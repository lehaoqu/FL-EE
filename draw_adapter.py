import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import argparse
import os

plt.rcParams['axes.prop_cycle']
matplotlib.rcParams['font.family'] = 'Times New Roman'

RATIO=1.7

MARKER_SIZE = 10/RATIO
TEXT_SIZE = 14/RATIO

LINE_WIDTH = 2/RATIO
MARKER_EDGE_WITH=2/RATIO

GRAY = '#777777'
DARK_GRAY = '#333333'
PURPLE = '#988ED5'
BROWN = '#8C6D31'
DARK_GREEN = '#467821'
DEEP_DARK_BLUE = '#253494'
RED = '#FF0000'

LIGHT_DARK_GRAY = '#A6A6A6'  # Lightened version of DARK_GRAY
LIGHT_GRAY = '#D3D3D3'  # Lightened version of GRAY
LIGHT_PURPLE = '#D6CDEA'  # Lightened version of PURPLE
LIGHT_BROWN = '#D6B37E'  # Lightened version of BROWN
LIGHT_GREEN = '#A8D7A3'  # Lightened version of DARK_GREEN
LIGHT_BLUE = '#A3B9E1'  # Lightened version of DEEP_DARK_BLUE
LIGHT_RED = '#FF9999'  # Lightened version of RED

RATIO=1.7

MARKER_SIZE = 10/RATIO
TEXT_SIZE = 14/RATIO
COLORS = [GRAY, DARK_GRAY, PURPLE, BROWN, DARK_GREEN, DEEP_DARK_BLUE, RED]
LIGHT_COLORS = [LIGHT_GRAY, LIGHT_DARK_GRAY, LIGHT_PURPLE, LIGHT_BROWN, LIGHT_GREEN, LIGHT_BLUE, LIGHT_RED]


COLOR=[LIGHT_RED, RED, BROWN, DARK_GREEN]
STYLE=['-', '--', '-.', ':']


def draw(data, path, title, x_label, y_label, y_range=(), x_range=(),y_step=0.1, x_step=1, suffix=''):
    fig, ax = plt.subplots()
    for idx, student in enumerate(data):
        # if model_name == 'scalefl' or model_name == 'exclusivefl':
        #     continue
        student = student[:200]
        step = int(len(student) // 50)
        student = student[::step]
        x = [i*step for i in range(len(student))]
        y = student
        
        plt.scatter(x, y, color=COLOR[idx], label=idx)


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
    plt.legend(loc='lower right')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 显示图表
    plt.show()
    
    o_dir = suffix+'/adapter/'
    t_dir = 'imgs/adapter/'
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)    
    plt.savefig(o_dir+path+'.png', dpi=300)
    plt.savefig(t_dir+path+'.png', dpi=300)
      

path = 'exps/test/darkflpg/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_distance.json'

t2s_list = []
with open(path, 'r') as f:
    t2s_list = json.load(f)


t2s_dct = {}
for item in t2s_list:
    for key, value in item.items():
        t2s_dct.setdefault(key, []).append(value)

for key, value in t2s_dct.items():
    s_num = len(t2s_dct[key][0])
    
    npy = -(np.array(value))
    a = np.exp(npy) / np.sum(np.exp(npy), axis=1, keepdims=True)
    b = list(a)
    
    students = list(map(list, zip(*b)))
    
    draw(students, f'CIFAR_noniid1000_{key}', f'CIFAR100_Adapter_Weight_{key}', 'Round', 'Weight', suffix='exps/test/darkflpg/full_boosted/noniid1000', y_range=(0.1,0.4))
