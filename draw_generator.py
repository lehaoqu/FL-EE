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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from trainer.policy.boosted import Policy


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






args = args_parser()

seed = args.seed
seed=1117
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
classes = [0,34,50,97]

def darkflpg_draw(label, dir, title,  only_legend=False):
    DIFFS = ["Difficulty 1", "Difficulty 2", "Difficulty 3", "Difficulty 4"]
    # COLORS = ["#E0F7FA", "#B2EBF2", "#4DD0E1", "#00BFA5"]
    COLORS = [RED, PURPLE, DARK_GREEN, BROWN]
    args.diff_generator=True
    a = torch.load('exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_G_12.pth')
    g = Generator_LATENT(args)

    g.load_state_dict(a, strict=False)

    ALL=1000
    fig, ax = plt.subplots()
    if not os.path.exists(dir):
        os.makedirs(dir)
    g.to(0)

    y_input = torch.tensor([label]*ALL, dtype=torch.long).to(0)
    diff=[]
    ds = [1, 2, 3, 4]
    for d in ds:
        diff.extend([d]*int(ALL/len(ds)))
        
    # diffs = torch.tensor(diff, dtype=torch.float).resize(ALL, 1).to(0)
    # diffs = torch.tensor(np.linspace(0, 5, ALL), dtype=torch.float).resize(ALL,1).to(0)
    diffs = torch.tensor(diff, dtype=torch.float).resize(ALL,1).to(0)
    eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)

    imgs = g(y_input, eps, diffs)
    
    # model_path = 'exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
    # config_path = 'exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
    # model = load_model_eval(args, model_path, config_path)
    # model.to(0)
    # md = []
    # for b_idx in range(int(ALL/20)):
    #     batch = {}
    #     batch['pixel_values'] = imgs[b_idx*20:(b_idx+1)*20]
    #     batch['labels'] = y_input[b_idx*20:(b_idx+1)*20]
    #     exits_logits = model(**batch, is_latent=True)
    #     for i in range(len(exits_logits[0])):
    #         diff =  difficulty_measure([exits_logits[0][i]], batch['labels'][i], metric='loss')
    #         md.append(diff.cpu().item())
    # print(md)
    # exit(0)
    
    imgs = imgs.resize(ALL, 197*192)

    X = imgs.detach().cpu().numpy()
    X = StandardScaler().fit_transform(X)

    # 创建t-SNE对象，并指定降维后的维度为2
    tsne = TSNE(n_components=2, n_iter=350)

    # 对数据进行降维
    print(X.shape)
    result = tsne.fit_transform(X)

    # 可视化降维后的结果
    if only_legend:
        for idx, d in enumerate(ds):
            # plt.scatter(result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 0], result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 1], color=COLORS[idx], label=DIFFS[idx], alpha=0.5)
            plt.scatter([], [], color=COLORS[idx], label=DIFFS[idx], alpha=0.5)
        legend = plt.legend(fontsize=TEXT_SIZE, ncol=4)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'{dir}pg_legend.pdf',  bbox_inches='tight')
        return
    
    
    for idx, d in enumerate(ds):
        plt.scatter(result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 0], result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 1], color=COLORS[idx], label=DIFFS[idx], alpha=0.5)
        
    # plt.legend(loc='lower right')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', bottom=False, left=False)
    # plt.title(title)
    plt.show()
    plt.savefig(f'{dir}label{label}.pdf', bbox_inches='tight')


# for c in classes:  
#     darkflpg_draw(c, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {c}')
# darkflpg_draw(50, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {50}')
# darkflpg_draw(99, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {99}')


def darkflpa_draw(label, dir, title, only_legend=False):
    COLORS = [RED, PURPLE, DARK_GREEN, BROWN]
    args.diff_generator=False
    gs = []
    ALL=1000
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
    X = StandardScaler().fit_transform(X)

    # 创建t-SNE对象，并指定降维后的维度为2
    tsne = TSNE(n_components=2, n_iter=300)

    # 对数据进行降维
    print(X.shape)
    result = tsne.fit_transform(X)
    
    
    # 可视化降维后的结果
    if only_legend:
        for idx in range(4):
            # plt.scatter(result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 0], result[int(ALL/len(ds))*(idx):int(ALL/len(ds))*(idx+1), 1], color=COLORS[idx], label=DIFFS[idx], alpha=0.5)
            plt.scatter([], [], color=COLORS[idx], label=f'Generator {idx}', alpha=0.5)
        legend = plt.legend(fontsize=TEXT_SIZE, ncol=4)
        plt.axis('off')
        plt.savefig(f'{dir}pa_legend.pdf')
        return

    # 可视化降维后的结果
    for idx in range(4):
        plt.scatter(result[int(ALL/4)*(idx):int(ALL/4)*(idx+1), 0], result[int(ALL/4)*(idx):int(ALL/4)*(idx+1), 1], color=COLORS[idx], label=f'Generator {idx}', alpha=0.5)

    # plt.legend(loc='lower right')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', bottom=False, left=False)
    plt.tight_layout()
    # plt.title(title)
    plt.show()
    plt.savefig(f'{dir}label{label}.pdf',  bbox_inches='tight')
    
    
# darkflpa_draw(0, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {0}')    
# darkflpa_draw(50, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {50}')    
# darkflpa_draw(99, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {99}')


# darkflpg_draw(0, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {0}', only_legend=True)
# darkflpa_draw(0, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {0}', only_legend=True)

for c in classes:
    # if c == 97: continue
    # darkflpg_draw(c, './imgs/generator/darkflpg/', f'Pseudo latent Visualization Label: {c}')
    darkflpa_draw(c, './imgs/generator/darkflpa/', f'Pseudo latent Visualization Label: {c}')  