import torch
import torch.nn as nn
from trainer.generator.generator import Generator_CIFAR, Generator_LATENT
import numpy as np
from PIL import Image
from utils.modelload.modelloader import load_model_eval
from utils.options import args_parser
import torch.nn.functional as F
import random
import math
from utils.train_utils import calc_target_probs
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy, difficulty_measure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
# COLORS = ["#E0F7FA", "#B2EBF2", "#4DD0E1", "#00BFA5"]


DIFFS = ["Easy", "Moderate", "Challenging", "Difficult"]



args = args_parser()

seed = args.seed
seed=1117
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def darkflpg_draw(label, dir, title):
    COLORS = ["#E0F7FA", "#B2EBF2", "#4DD0E1", "#00BFA5"]
    args.diff_generator=True
    a = torch.load('exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_G_12.pth')
    g = Generator_LATENT(args)

    g.load_state_dict(a, strict=False)

    ALL=500
    fig, ax = plt.subplots()
    if not os.path.exists(dir):
        os.makedirs(dir)
    g.to(0)

    y_input = torch.tensor([label]*ALL, dtype=torch.long).to(0)
    diff=[]
    ds = [0, 1.5, 3, 4.5]
    for d in ds:
        diff.extend([d]*int(ALL/len(ds)))
        
    # diffs = torch.tensor(diff, dtype=torch.float).resize(ALL, 1).to(0)
    diffs = torch.tensor(np.linspace(0, 5, ALL), dtype=torch.float).resize(ALL,1).to(0)
    eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)

    imgs = g(y_input, eps, diffs)
    imgs = imgs.resize(ALL, 197*192)

    X = imgs.detach().cpu().numpy()

    # 创建t-SNE对象，并指定降维后的维度为2
    tsne = TSNE(n_components=2)

    # 对数据进行降维
    print(X.shape)
    result = tsne.fit_transform(X)

    # 可视化降维后的结果
    for idx, d in enumerate(ds):
        plt.scatter(result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 0], result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 1], color=COLORS[idx], label=DIFFS[idx])

    plt.legend(loc='lower right')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.show()
    plt.savefig(f'{dir}label{label}.png')


darkflpg_draw(0, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {0}')
darkflpg_draw(50, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {50}')
darkflpg_draw(99, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {99}')


def darkflpa_draw(label, dir, title):
    COLORS = [RED, DARK_GREEN, BROWN, PURPLE]
    args.diff_generator=False
    gs = []
    ALL=64
    for i in range(4):
        a = torch.load(f'exps/BASE_CIFAR/full_boosted/noniid1000/darkflpa2_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_G_{i}.pth')
        g = Generator_LATENT(args)
        g.load_state_dict(a, strict=False)
        g.to(0)
        gs.append(g)
    
    fig, ax = plt.subplots()
    if not os.path.exists(dir):
        os.makedirs(dir)

    
    imgs = []
    for i in range(4):
        y_input = torch.tensor([label]*int(ALL/4), dtype=torch.long).to(0)
        eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)
        imgs.append(gs[i](y_input, eps))
    
    imgs = torch.cat(imgs, dim=0) 
    
    imgs = imgs.resize(ALL, 197*192)

    X = imgs.detach().cpu().numpy()

    # 创建t-SNE对象，并指定降维后的维度为2
    tsne = TSNE(n_components=2)

    # 对数据进行降维
    print(X.shape)
    result = tsne.fit_transform(X)

    # 可视化降维后的结果
    for idx in range(4):
        plt.scatter(result[int(ALL/4)*(idx):int(ALL/4)*(idx+1), 0], result[int(ALL/4)*(idx):int(ALL/4)*(idx+1), 1], color=COLORS[idx], label=f'Generator {idx}')

    plt.legend(loc='lower right')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.show()
    plt.savefig(f'{dir}label{label}.png')
    
    
darkflpa_draw(0, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {20}')    
darkflpa_draw(50, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {50}')    
darkflpa_draw(99, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {99}')