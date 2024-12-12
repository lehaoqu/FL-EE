import matplotlib.pyplot as plt
import numpy as np
import json
from utils.options import args_parser
import os


if __name__ == '__main__':
    args = args_parser()
    eval_dir = args.suffix


    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    # name - > dict{'test':[], 'flops':[]}
    data = {}    
    
    for model_path in model_paths:
        if '_eval' in model_path:
            print(model_path)
            
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0]
