import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import argparse
import os
import math
import re
from scipy.interpolate import make_interp_spline

plt.rcParams['axes.prop_cycle']
matplotlib.rcParams['font.family'] = 'Times New Roman'

RATIO=1.7

LINE_WIDTH = 2/RATIO
MARKER_EDGE_WITH=2/RATIO
MARKER_SIZE = 10/RATIO
TEXT_SIZE = 40/RATIO
TRICKS_SIZE = 22/RATIO
TOTAL_FLOPS = 3.45271/4


BROWN = '#8C6D31'
RED = '#FF0000'
LIGHT_RED = '#FF9999'  # Lightened version of RED
YELLOW  = '#FFCC5B'
GREEN   = '#3FB11D'
BLUE    = '#4DD0FD'
PURPLE  = '#BF00BF'
LIGHT_GARY = '#D3D3D3'


COLOR={'darkflpa2':LIGHT_RED, 'darkflpg': RED, 'eefl': BROWN, 'depthfl': YELLOW, 'reefl': GREEN, 'inclusivefl': BLUE, 'scalefl': PURPLE, 'exclusivefl': BROWN}
MARKER={'darkflpa2':'v', 'darkflpg': 'v', 'eefl':'s', 'depthfl':'s', 'reefl': 'o', 'inclusivefl': '^', 'scalefl': 'D', 'exclusivefl': 'D'}
STYLE={'darkflpa2':'-', 'darkflpg': '-', 'eefl':'--', 'depthfl':'--', 'reefl': '--', 'inclusivefl': '--', 'scalefl': '--', 'exclusivefl': '--'}
NAMES = {'darkflpa2':'DarkDistill+', 'darkflpg': 'DarkDistill', 'eefl':'EEFL', 'depthfl':'DepthFL', 'reefl': 'ReeFL', 'inclusivefl': 'InclusiveFL', 'scalefl': 'ScaleFL', 'exclusivefl': 'ExclusiveFL'}
APPS = ['inclusivefl', 'scalefl', 'depthfl', 'reefl', 'darkflpg', 'darkflpa2']

def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('--suffix', type=str, default='dir')
    return parser.parse_args()


def anytime_bar(data, path, x_label, y_label, title='', y_range=(), x_range=(),y_step=1, x_step=1, suffix='', y_none=False):
    fig, ax = plt.subplots()
    plt.grid(color='white', linestyle='-', linewidth=0.5, axis='y', zorder=0)
    # 设置柱状图的宽度
    bar_width = 0.11

    # 设置每个组的位置
    group_labels = ['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4']
    index = np.arange(len(group_labels))
    cnt=0
    for model_name in APPS:
        if 'LORA' in path:
            if model_name == 'darkflpa2':
                continue
        y = data[model_name]
        plt.bar(index + cnt * bar_width, y, bar_width, label=NAMES[model_name], color=COLOR[model_name], zorder=2, edgecolor='white')
        cnt+=1
    
    if len(y_range) == 2:
        plt.ylim(*y_range)
        if y_step > 0:
            EPS = 1e-6
            plt.yticks(np.arange(y_range[0], y_range[1] + EPS, y_step),)    
    
    # 添加标签和标题
    # plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontsize=TEXT_SIZE)

    plt.xticks(index + int(cnt/2) * bar_width, group_labels)

    # 添加图例
    plt.legend(ncol=3, loc="lower center")
    plt.gca().set_facecolor('#EAEAF2')

    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='both', which='major', labelsize=TRICKS_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 显示图表
    plt.tight_layout()
    plt.show()
    plt.savefig('')
    o_dir = suffix+'/anytime/'
    t_dir = 'imgs/anytime/'
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)    
    plt.savefig(o_dir+path+'.pdf')
    plt.savefig(t_dir+path+'.pdf')


def budget(data, path, title, x_label, y_label, y_range=(), x_range=(),y_step=1, x_step=1, suffix='', y_none=False):
    fig, ax = plt.subplots()
    
    for model_name in APPS:
        if 'LORA' in path:
            if model_name == 'darkflpa2':
                continue
        # if model_name == 'scalefl' or model_name == 'exclusivefl':
        #     continue
        x = data[model_name]['flops'][5:25] +  data[model_name]['flops'][25::2]
        y = data[model_name]['test'][5:25] + data[model_name]['test'][25::2]
        
        x = [TOTAL_FLOPS * i for i in x]
        # x = data[model_name]['flops']
        # y = data[model_name]['test']
        # print(x)
        # spl = make_interp_spline(x,y)
        # x_smooth = np.linspace(min(x), max(x), 200)
        # y_smooth = spl(x_smooth)
        
        plt.plot(x, y, color=COLOR[model_name], label=NAMES[model_name], marker=MARKER[model_name], linestyle=STYLE[model_name], markeredgecolor='white', markeredgewidth=1)


    if len(y_range) == 2:
        plt.ylim(*y_range)
        if y_step > 0:
            EPS = 1e-6
            plt.yticks(np.arange(y_range[0], y_range[1] + EPS, y_step),)    
    

    if len(x_range) == 2:
        plt.xlim(*x_range)
        if x_step > 0:
            EPS = 1e-6
            plt.xticks(np.arange(x_range[0], x_range[1] + EPS, x_step),)    
    plt.legend(ncol=3, loc="lower center")

    # plt.title(title, fontsize=TEXT_SIZE)
    if y_none is True:
        ax.tick_params(axis='both', which='both', left=False, labelleft=False)
    else:
        plt.ylabel(y_label, fontsize=TEXT_SIZE)
    
    plt.xlabel(x_label, fontsize=TEXT_SIZE)   
    
    plt.gca().set_facecolor('#EAEAF2')
    plt.grid(color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='both', which='major', labelsize=TRICKS_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    o_dir = suffix+'/budget/'
    t_dir = 'imgs/budget/'
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)    
    plt.savefig(o_dir+path+'.pdf')
    plt.savefig(t_dir+path+'.pdf')
        
def round(data, path, title, x_label, y_label, y_range=(), x_range=(),y_step=1, x_step=1, suffix=''):
    fig, ax = plt.subplots()
    

    for model_name in APPS:
        # if model_name == 'scalefl' or model_name == 'exclusivefl':
        #     continue
        x = [i*10 for i in range(50)]
        y = data[model_name]
        if 'loss' in title and 'scalefl' in model_name:
            y = [l*4 for l in y]
        plt.plot(x, y, color=COLOR[model_name], label=NAMES[model_name], linestyle=STYLE[model_name])


    if len(y_range) == 2:
        plt.ylim(*y_range)
        if y_step > 0:
            EPS = 1e-6
            plt.yticks(np.arange(y_range[0], y_range[1] + EPS, y_step), fontsize=TEXT_SIZE)    
    

    if len(x_range) == 2:
        plt.xlim(*x_range)
        if x_step > 0:
            EPS = 1e-6
            plt.xticks(np.arange(x_range[0], x_range[1] + EPS, x_step), fontsize=TEXT_SIZE)    
    if 'loss' in title:
        plt.legend(loc='upper right')
    else:
        plt.legend(loc='lower right')

    # plt.title(title, fontsize=TEXT_SIZE)
    plt.xlabel(x_label, fontsize=TEXT_SIZE)
    plt.ylabel(y_label, fontsize=TEXT_SIZE)
    
    plt.gca().set_facecolor('#EAEAF2')
    plt.grid(color='white', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='both', which='major', labelsize=TRICKS_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    if 'Acc' in y_label:
        o_dir = suffix+'/acc/'
        t_dir = 'imgs/acc/'
    elif 'Loss' in y_label:
        o_dir = suffix+'/loss/'
        t_dir = 'imgs/loss/'
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)    
    plt.savefig(o_dir+path+'.pdf')
    plt.savefig(t_dir+path+'.pdf')
        
    

def cifar_Full_1000():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid1000_Full', title=f'CIFAR100_noniid1000_Full', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        y_range=(60, 72),
        y_step=2,
        #  y_range=(55, 72),
        #  y_step=5,
         x_range=(1.0, 3.5),
         x_step=0.5,
         suffix=suffix,
         )
    
def cifar_LORA_1000():
    suffix = 'exps/BASE_CIFAR/lora_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid1000_LORA', title=f'CIFAR100_noniid1000_LORA', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        y_range=(60, 70),
        y_step=1,
        x_range=(1.5, 3.5),
        x_step=0.5,
        # y_range=(54, 70),
        # y_step=2,
         suffix=suffix
         )
  
def cifar_Full_1():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid1_Full', title=f'CIFAR100_noniid1_Full', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        # y_range=(66, 70.5),
        # x_range=(1.8, 4.0),
        y_range=(50, 72),
        y_step=2,
        suffix=suffix,
        )
    
def cifar_LORA_1():
    suffix = 'exps/BASE_CIFAR/lora_boosted/noniid1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid1_LORA', title=f'CIFAR100_noniid1_LORA', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
         y_range=(60, 70),
         x_range=(1.5, 3.5),
        x_step=0.5,
        y_step=1,
        # y_range=(54, 70),
        #  y_step=2,
         suffix=suffix
         )
  
def cifar_Full_01():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid0.1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid0.1_Full', title=f'CIFAR100_noniid0.1_Full', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
         y_range=(50, 66),
         y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix,
         )
    
def cifar_LORA_01():
    suffix = 'exps/BASE_CIFAR/lora_boosted/noniid0.1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path=f'CIFAR100_noniid0.1_LORA', title=f'CIFAR100_noniid0.1_LORA', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
         y_range=(56, 66),
         y_step=1,
         x_range=(1.5, 3.5),
         x_step=0.5,
         suffix=suffix
         )
   

  
def svhn_Full():
    suffix = 'exps/BASE_SVHN/full_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path='SVHN_Full', title='SVHN_Full', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        y_range=(87.5, 89.5),
        y_step=0.5,
        #  x_range=(2.2, 4.0),
        #  x_step=0.5,
         suffix=suffix
         ) 
    
def svhn_LORA():
    suffix = 'exps/BASE_SVHN/lora_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path='SVHN_LORA', title='SVHN_LORA', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        #  y_range=(83, 88),
        #  x_range=(2.2, 4.0),
        #  x_step=0.5,
         suffix=suffix
         )
           
def speechcmds_Full():
    suffix = 'exps/BASE_SPEECHCMDS/full_boosted'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path='SpeechCmds_Full', title='SpeechCmds_Full', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        # y_range=(91, 94),
        #  y_step=0.5,
        #  x_range=(2.2, 4.0),
        #  x_step=0.5,
         suffix=suffix
         ) 
    
def speechcmds_LORA():
    suffix = 'exps/BASE_SPEECHCMDS/lora_boosted'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_eval' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    budget(data, path='SpeechCmds_LORA', title='SpeechCmds_LORA', x_label='Average budget (in MUL-ADD) $\\times 10^{10}$', y_label='Accuracy (%)',
        y_range=(90, 92),
        #  x_range=(2.2, 4.0),
        #  x_step=0.5,
         suffix=suffix
         )       



def cifar_Full_acc_1000():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_acc' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid1000_Full', title=f'CIFAR100_noniid1000_Full_acc', x_label='Round', y_label='Accuracy (%)',
        #  y_range=(54, 66),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def cifar_Full_acc_1():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_acc' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid1_Full', title=f'CIFAR100_noniid1_Full_acc', x_label='Round', y_label='Accuracy (%)',
        #  y_range=(50, 66),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def cifar_Full_acc_01():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid0.1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_acc' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid0.1_Full', title=f'CIFAR100_noniid0.1_Full_acc', x_label='Round', y_label='Accuracy (%)',
        #  y_range=(54, 62),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def svhn_Full_acc():
    suffix = 'exps/BASE_SVHN/full_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_acc' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path='SVHN_Full', title='SVHN_Full_acc', x_label='Round', y_label='Accuracy (%)',
         y_range=(90, 95),
         y_step=2,
         suffix=suffix
         )

def speechcmds_Full_acc():
    suffix = 'exps/BASE_SPEECHCMDS/full_boosted'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_acc' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path='SpeechCmds_Full', title='SpeechCmds_Full_acc', x_label='Round', y_label='Accuracy (%)',
         y_range=(90.5, 93.6),
         y_step=0.5,
         suffix=suffix
         )



def cifar_Full_loss_1000():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_loss' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid1000_Full', title=f'CIFAR100_noniid1000_Full_loss', x_label='Round', y_label='Loss',
        #  y_range=(54, 66),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def cifar_Full_loss_1():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_loss' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid1_Full', title=f'CIFAR100_noniid1_Full_loss', x_label='Round', y_label='Loss',
        #  y_range=(50, 66),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def cifar_Full_loss_01():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid0.1'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_loss' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path=f'CIFAR100_noniid0.1_Full', title=f'CIFAR100_noniid0.1_Full_loss', x_label='Round', y_label='Loss',
        #  y_range=(54, 62),
        #  y_step=2,
        #  x_range=(1.6, 4.0),
         suffix=suffix
         )

def svhn_Full_loss():
    suffix = 'exps/BASE_SVHN/full_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_loss' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path='SVHN_Full', title='SVHN_Full_loss', x_label='Round', y_label='Loss',
        #  y_range=(90, 95),
        #  y_step=2,
         suffix=suffix
         )

def speechcmds_Full_loss():
    suffix = 'exps/BASE_SPEECHCMDS/full_boosted'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '_loss' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path+'.json', 'r') as f:
                    data[name_without_extension] = json.load(f)
    round(data, path='SpeechCmds_Full', title='SpeechCmds_Full_loss', x_label='Round', y_label='Loss',
        #  y_range=(90.5, 93.6),
        #  y_step=0.5,
         suffix=suffix
         )


def cifar_Full_1000_any():
    suffix = 'exps/BASE_CIFAR/full_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'CIFAR100_noniid1000_Full', x_label='Exits', y_label='Accuracy (%)',
        y_range=(40, 72),
        # y_step=2,
        #  y_range=(55, 72),
        #  y_step=5,
        #  x_range=(1.6, 4.0),
         suffix=suffix,
         )
 
def cifar_LORA_1000_any():
    suffix = 'exps/BASE_CIFAR/lora_boosted/noniid1000'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'CIFAR100_noniid1000_LORA', x_label='Exits', y_label='Accuracy (%)',
         suffix=suffix,
         )
 
def svhn_Full_any():
    suffix = 'exps/BASE_SVHN/full_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'SVHN_Full', x_label='Exits', y_label='Accuracy (%)',
         suffix=suffix,
         y_range=(80, 90)
         )

def svhn_LORA_any():
    suffix = 'exps/BASE_SVHN/lora_boosted/noniid'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'SVHN_LORA', x_label='Exits', y_label='Accuracy (%)',
         suffix=suffix,
         y_range=(80, 90)
         )

def speechcmds_Full_any():
    suffix = 'exps/BASE_SPEECHCMDS/full_boosted/'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'SPEECHCMDS_FULL', x_label='Exits', y_label='Accuracy (%)',
         suffix=suffix,
         y_range=(85, 95)
         )

def speechcmds_LORA_any():
    suffix = 'exps/BASE_SPEECHCMDS/lora_boosted/'
    eval_dir = suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')) for f in file_names if 'eval.txt' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    
    data = {}    
    for model_path in model_paths:
        if '.txt' in model_path:
            # print(model_path)
            base_name = os.path.basename(model_path)
            name_without_extension = os.path.splitext(base_name)[0].split('_')[0]
            print(name_without_extension)
            if name_without_extension != 'eefl' and name_without_extension != 'exclusivefl':
                with open(model_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if 'avg' in line:
                            numbers = re.findall(r'\d+\.\d+', line)[:-1]
                            # 将提取的字符串转换为浮点数，并存储在列表中
                            data_list = [float(num) for num in numbers]
                            data[name_without_extension] = data_list
    anytime_bar(data, path=f'SPEECHCMDS_LORA', x_label='Exits', y_label='Accuracy (%)',
         suffix=suffix,
         y_range=(85, 95)
         )


## == ANYTIME ==
# cifar_Full_1000_any()
# cifar_LORA_1000_any()
# svhn_Full_any()
# svhn_LORA_any()
# speechcmds_Full_any()
# speechcmds_LORA_any()

# # #== BUDGET ==
cifar_Full_1000()    
cifar_LORA_1000()
cifar_Full_1()    
cifar_LORA_1()
cifar_Full_01()    
cifar_LORA_01()

# svhn_Full()
# svhn_LORA()

# speechcmds_Full()
# speechcmds_LORA()
    

# # = ACC ==
# cifar_Full_acc_1000()
# cifar_Full_acc_1()
# cifar_Full_acc_01()
# # # 不太行
# svhn_Full_acc()
# speechcmds_Full_acc()


# # # == LOSS ==
# cifar_Full_loss_1000()
# cifar_Full_loss_1()
# cifar_Full_loss_01()
# svhn_Full_loss()
# speechcmds_Full_loss()