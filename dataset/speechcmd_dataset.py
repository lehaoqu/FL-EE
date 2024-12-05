import torch
import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset
from dataset.utils.dataset_utils import load_np, load_pkl, load_mat
import os
import random
from typing import Tuple
from pathlib import Path

import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms as tv_transforms
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS, HASH_DIVIDER, EXCEPT_FOLDER, _load_list
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio import transforms, load
from torchaudio.compliance.kaldi import fbank

from collections import defaultdict
import flwr #! DON'T REMOVE -- bad things happen
import pdb
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)
BKG_NOISE = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav", "running_tap.wav", "white_noise.wav"]
CLASSES_12 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']


class SPEECHCOMMANDSDATASET(Dataset):
    def __init__(self, root, type):
        super().__init__()
        self.wav2fbank = True
        self.set = SPEECHCOMMANDS(root=root, download=True, subset=type)
        self.classes_to_use = [Path(f.path).name for f in os.scandir(self.set._path) if f.is_dir() and f.name != EXCEPT_FOLDER and not f.name.isdigit()]
        self.get()


    def pad_sequence(self, batch): #! borrowed from [1]
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0,2,1)


    def label_to_index(self, word, labels): #! borrowed from [1]
        # Return the position of the word in labels
        if word in labels:
            return torch.tensor(labels.index(word))
        else:
            return torch.tensor(10) # higlight as `unknown`

    # 定义整理函数，用于整理数据批次
    def collate_fn(self, batch): #! ~borrowed from [1]
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, sr, label, *_ in batch:

            tensors += [waveform]
            targets += [self.label_to_index(label, self.classes_to_use)]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        # ])
        # tensors = transform(tensors)
        return tensors, targets
    
    def _wav2fbank(self, waveform, sr, mel_bins=128, target_length=128):
        # eavily borrowing from `make_features()` in: https://colab.research.google.com/github/YuanGongND/ast/blob/master/Audio_Spectrogram_Transformer_Inference_Demo.ipynb#scrollTo=sapXfOwbhrzG
        f_bank = fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
            frame_shift=10)

        n_frames = f_bank.shape[0]

        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            f_bank = m(f_bank)
        elif p < 0:
            f_bank = f_bank[0:target_length, :]

        f_bank = (f_bank - (-4.2677393)) / (4.5689974 * 2)

        return f_bank
    
    def get(self):
        self.pixel_values = []
        self.labels = []
        for data in self.set:
            waveform, sr, label, *_ = data
             
            if self.wav2fbank:
                self.pixel_values += [self._wav2fbank(waveform, sr)]
                # print(tensors[-1].shape)
            else:
                self.pixel_values += [waveform]
             
            self.labels += [self.label_to_index(label, self.classes_to_use)]
        self.pixel_values = [sample for sample in self.pad_sequence(self.pixel_values)]
        self.labels = [sample for sample in torch.stack(self.labels)]
    
    def __len__(self):
        return len(self.set) 



class SPEEDCMDSClassificationDataset(Dataset):
    def transform_for_vit(images: torch.tensor):

        transform = tv_transforms.Compose([
                tv_transforms.Resize((224, 224)),
            ])
        
        images = images.float()
        images = images.unsqueeze(1)        
        tf = torch.stack([transform(image) for image in images], dim=0)
        return tf.repeat(1,3,1,1)
    
    
    def generator_transform_tensor(images: torch.tensor):
        
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])
        device = images.device
        images_reshaped = images.view(-1, 3, 32, 32)
        return torch.stack([transform(image) for image in images_reshaped], dim=0)
    
    
    def __init__(self, args=None, path=None, eval_valids=False):
        self.path = path
        if eval_valids:
            dict_all = [load_pkl(f'{path}{i}.pkl') for i in range(args.total_num)]
            total_data = {}
            for key in dict_all[0].keys():
                for dic in dict_all:
                    total_data.setdefault(key, []).extend(dic[key])
            self.ann = total_data
            
        else:
            self.ann = load_pkl(path)
            
            self.ann[b'data'] = [torch.tensor(row, dtype=torch.float32) for row in self.ann[b'data']]
            self.ann[b'fine_labels'] = [torch.tensor(row, dtype=torch.long) for row in self.ann[b'fine_labels']]
            
        self.pixel_values = self.ann[b'data']
        self.labels = self.ann[b'fine_labels']
        
        
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):
        return dict(
            pixel_values=self.pixel_values[index],
            labels=self.labels[index],
        )

