from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
from dataset.utils.dataset_utils import load_np, load_pkl

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
import torch
from tqdm import tqdm


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.pixel_values = []
        self.labels = []

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in tqdm(list_of_dirs):
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])

                        self.images.append(item)
                        img_path, t = item
                        with open(img_path, 'rb') as f:
                            sample = Image.open(img_path)
                            sample = sample.convert('RGB')
                        if self.transform is not None:
                            sample = self.transform(sample).float()
                        self.pixel_values.append(sample)
                        t = torch.tensor(t, dtype=torch.long)
                        self.labels.append(t)
                        

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample).float()

        return dict(
            pixel_values=sample,
            labels=torch.tensor(tgt, dtype=torch.long),
        )


class TinyImageNetClassificationDataset(Dataset):
    def transform_for_vit(images: torch.tensor):

        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])
        
        images = images.float()
        images_reshaped = images.view(-1, 3, 64, 64)
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
