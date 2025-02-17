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
DARK_GREEN = '#467821'


COLOR={'darkflpa2':LIGHT_RED, 'darkflpg': RED, 'eefl': BROWN, 'depthfl': YELLOW, 'reefl': GREEN, 'inclusivefl': BLUE, 'scalefl': PURPLE, 'exclusivefl': BROWN}
MARKER={'darkflpa2':'s', 'darkflpg': 's', 'eefl':'s', 'depthfl':'v', 'reefl': 'o', 'inclusivefl': '^', 'scalefl': 'D', 'exclusivefl': 'D'}
STYLE={'darkflpa2':'-', 'darkflpg': '-', 'eefl':'--', 'depthfl':'--', 'reefl': '--', 'inclusivefl': '--', 'scalefl': '--', 'exclusivefl': '--'}
NAMES = {'darkflpa2':'DarkDistill-PL', 'darkflpg': 'DarkDistill', 'eefl':'EEFL', 'depthfl':'DepthFL', 'reefl': 'ReeFL', 'inclusivefl': 'InclusiveFL', 'scalefl': 'ScaleFL', 'exclusivefl': 'ExclusiveFL'}
APPS = ['inclusivefl', 'scalefl', 'depthfl', 'reefl', 'darkflpg', 'darkflpa2']



def budget():
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))
    for model_name in APPS:

        plt.plot([], [], color=COLOR[model_name], label=NAMES[model_name], marker=MARKER[model_name], linestyle=STYLE[model_name], markeredgecolor='white', markeredgewidth=1)


    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.tick_params(axis='y', which='both', labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.legend(ncol=6, loc="lower center")

  
    # 显示图表
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    plt.savefig('imgs/over_legend.pdf')



def round():
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))

    for model_name in APPS:

            
        plt.plot([], [], color=COLOR[model_name], label=NAMES[model_name], linestyle=STYLE[model_name])

    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.tick_params(axis='y', which='both', labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.legend(ncol=6, loc="lower center")

  
    # 显示图表
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    plt.savefig('imgs/over_round_legend.pdf')
  
        
    
def darkflpg_draw():
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 6))
        
    DIFFS = ["Difficulty 1", "Difficulty 2", "Difficulty 3", "Difficulty 4"]
    # COLORS = ["#E0F7FA", "#B2EBF2", "#4DD0E1", "#00BFA5"]
    COLORS = [RED, PURPLE, DARK_GREEN, BROWN]
    
    
    for idx in range(4):
        plt.scatter([], [], color=COLORS[idx], label=DIFFS[idx], alpha=0.5)
        
    ax.tick_params(axis='x', which='both', top=False, bottom=False, length=0)
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)
    plt.tick_params(axis='x', which='both', labelbottom=False)
    plt.tick_params(axis='y', which='both', labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.legend(ncol=4, loc="lower center")

  
    # 显示图表
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('imgs/G_darkflpg_legend.pdf')



# budget()
round()
# darkflpg_draw()