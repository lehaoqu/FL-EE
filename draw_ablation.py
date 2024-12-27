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
TOTAL_FLOPS = 3.45271/4


A = '#8B33B9'
B = '#C256A5'
C = '#E88580'

pa_A = 'DarkDistill+'
pa_B = 'DarkDistill+ w/o DI'
pa_C = 'DarkDistill+ w/o RKD'

pg_A = 'DarkDistill'
pg_B = 'DarkDistill w/o DC'
pg_C = 'DarkDistill w/o RKD'




COLOR={'DarkDistill+': A, 'DarkDistill+ w/o DI':B, 'DarkDistill+ w/o RKD':C, 'DarkDistill': A, 'DarkDistill w/o DC':B, 'DarkDistill w/o RKD':C}
APPS_PA = ['DarkDistill+', 'DarkDistill+ w/o DI', 'DarkDistill+ w/o RKD']
APPS_PG = ['DarkDistill', 'DarkDistill w/o DC', 'DarkDistill w/o RKD']

def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('--suffix', type=str, default='dir')
    return parser.parse_args()


def ablation_bar(data, path, y_label, title='', y_range=(), x_range=(),y_step=1, x_step=1, suffix='', y_none=False):
    fig, ax = plt.subplots()
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14    
    plt.grid(color='white', linestyle='-', linewidth=0.5, axis='y', zorder=0)
    # 设置柱状图的宽度
    bar_width = 0.2

    # 设置每个组的位置
    group_labels = ['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4']
    index = np.arange(len(group_labels))
    cnt=0
    if 'PA' in path:
        APPS = APPS_PA
    else:
        APPS = APPS_PG
    for model_name in APPS:
        y = data[model_name]
        plt.bar(index + cnt * bar_width, y, bar_width, label=model_name, color=COLOR[model_name], zorder=2, edgecolor='white')
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
    plt.legend(ncol=1, loc="upper left")
    plt.gca().set_facecolor('#EAEAF2')

    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='both', which='major', labelsize=TEXT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 显示图表
    plt.tight_layout()
    plt.show()
    plt.savefig('')

    t_dir = 'imgs/ablation/'

    if not os.path.exists(t_dir):
        os.makedirs(t_dir)    

    plt.savefig(t_dir+path+'.pdf')
    

def pa_full_1000():
    data = {
        pa_A:[53.36,65.78,71.39,71.43],
        pa_B:[53.80,65.26,70.54,70.30],
        pa_C:[53.67,64.89,70.54,70.30]
    }
    
    ablation_bar(data, 'PA_CIFAR_FULL_1000', y_label='Accuracy (%)', y_range=(50,72), y_step=5)
 
    
def pg_full_1000():
    data = {
        pg_A:[54.54, 66.14, 71.37, 71.13],
        pg_B:[53.40, 65.67, 70.96, 70.60],
        pg_C:[53.67, 64.89, 70.54, 70.30]
    }
    
    ablation_bar(data, 'PG_CIFAR_FULL_1000', y_label='Accuracy (%)', y_range=(50,72), y_step=5)


def pg_lora_1000():
    data = {
        pg_A:[45.29, 62.68, 69.42, 69.52],
        pg_B:[44.79, 62.51, 69.18, 69.22],
        pg_C:[44.84, 62.19, 68.33, 68.50]
    }
    
    ablation_bar(data, 'PG_CIFAR_LORA_1000', y_label='Accuracy (%)', y_range=(40,70), y_step=5)



pa_full_1000()
pg_full_1000()
pg_lora_1000()