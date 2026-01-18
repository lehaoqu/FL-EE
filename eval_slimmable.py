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
from utils.train_utils import fuse_curves_take_max, area_under_fitted_curve


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
            slim_x_list = {}
            slim_y_list = {}
            # eval loop for each slim ratio
            for ratio in slim_ratios:
                # print(f"Evaluating at slim ratio: {ratio}")
                if full_model.config.slimmable:
                    from utils.modelload.slimmable import set_width_ratio
                    
                    eval._log((f'eval model:{os.path.basename(model_path)}').center(80, '='))
                    eval._log(f'Setting width ratio to {ratio}')
                    set_width_ratio(ratio, full_model)
                    eval.eval(model_path+'.pth', model_path+'.json', model=full_model)

                    eval_path = eval.eval_json_path
                    dct = json.loads(open(eval_path, 'r').read())
                    x = dct['flops']
                    y = dct['test']
                    slim_x_list[ratio] = x
                    slim_y_list[ratio] = y
            
            fx, fy = fuse_curves_take_max(slim_x_list, slim_y_list, ref_ratio=1.0)
            area, acc = area_under_fitted_curve(fy, fx)
            print(f"AUC: {area}, Budgeted Acc: {acc}")
            ratio_1_eval_path = eval.eval_dir+eval.model_path+f'_slim_1.0_eval.json'
            ratio_1_dct = json.loads(open(ratio_1_eval_path, 'r').read())
            json.dumps({'budgeted_acc': acc, **{k: v for k, v in ratio_1_dct.items()}}, open(ratio_1_eval_path, 'w'))
            