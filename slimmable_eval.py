import os
import torch
import torch.nn as nn
import math
import importlib
from PIL import Image
import numpy as np
import argparse
import json

from tqdm import tqdm
from transformers import BertTokenizer

from utils.options import args_parser
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset
from dataset.imagenet_dataset import TinyImageNetClassificationDataset
from dataset.speechcmd_dataset import SPEEDCMDSClassificationDataset
from eval import *


if __name__ == '__main__':
    args = args_parser()
    eval_dir = args.suffix
    args.img_dir = eval_dir + "/img"
    eval = Eval(args=args)

    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f and '.png' not in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    # print(model_paths)
    for model_path in model_paths:
        if  args.policy in model_path and 'G_' not in model_path and 'loss' not in model_path and 'acc' not in model_path and 'distance' not in model_path and 'budget' not in model_path:
            print(model_path)
            full_model = load_model_eval(args, model_path+'.pth', config_path=model_path+'.json')
            slim_ratios = full_model.config.slim_ratios if full_model.config.slimmable else [1.0]
            print(slim_ratios)
            # eval loop for each slim ratio
            for ratio in slim_ratios:
                print(f"Evaluating at slim ratio: {ratio}")
                if full_model.config.slimmable:
                    from utils.modelload.slimmable import set_width_ratio
                    
                    eval._log((f'eval model:{os.path.basename(model_path)}').center(80, '='))
                    eval._log(f'Setting width ratio to {ratio}')
                    set_width_ratio(ratio, full_model)
                    eval.eval(model_path+'.pth', model_path+'.json', model=full_model)