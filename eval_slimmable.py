import os
from random import random
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
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


def _build_exit_indices_by_ratio(eval_obj, slim_ratios, probs):
    """Collect exit sample indices for each slim ratio using dynamic thresholds."""
    exit_indices_by_ratio = {}
    tester = eval_obj.tester
    preds = eval_obj.test_exits_preds
    targets = eval_obj.test_targets

    flops = getattr(tester, 'flops', [float(i + 1) for i in range(tester.n_exits)])

    for ratio in slim_ratios:
        acc, exp_flops, T, exit_sample_indices = tester.dynamic_eval_find_threshold(
            preds, targets, probs, flops, return_exit_indices=True
        )
        exit_indices_by_ratio[str(ratio)] = {
            'acc': float(acc),
            'expected_flops': float(exp_flops),
            'T': [float(v) for v in T.detach().cpu().tolist()],
            'exit_sample_indices': exit_sample_indices,
            'exit_counts': [len(lst) for lst in exit_sample_indices],
        }
    return exit_indices_by_ratio


def _plot_jaccard_matrices(exit_indices_by_ratio, out_path: str):
    if not exit_indices_by_ratio:
        return

    ratio_items = []
    for k in exit_indices_by_ratio.keys():
        try:
            ratio_items.append((float(k), k))
        except Exception:
            ratio_items.append((float('inf'), k))
    ratio_items.sort(key=lambda t: t[0])
    ratio_keys = [k for _v, k in ratio_items]

    first_key = ratio_keys[0]
    n_exits = len(exit_indices_by_ratio[first_key]['exit_sample_indices'])
    n_ratios = len(ratio_keys)

    exit_sets = []
    for rk in ratio_keys:
        sets_for_ratio = [set(lst) for lst in exit_indices_by_ratio[rk]['exit_sample_indices']]
        exit_sets.append(sets_for_ratio)

    cols = 2
    rows = int(math.ceil(n_exits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)

    for e in range(n_exits):
        jacc = np.zeros((n_ratios, n_ratios), dtype=float)
        for i in range(n_ratios):
            a = exit_sets[i][e]
            for j in range(n_ratios):
                b = exit_sets[j][e]
                if not a and not b:
                    jacc[i, j] = 1.0
                else:
                    jacc[i, j] = len(a & b) / max(1, len(a | b))

        ax = axes[e // cols][e % cols]
        ax.imshow(jacc, vmin=0.0, vmax=1.0, cmap='magma')
        ax.set_title(f'Exit {e}: Jaccard')
        ax.set_xticks(list(range(n_ratios)))
        ax.set_yticks(list(range(n_ratios)))
        ax.set_xticklabels([str(k) for k in ratio_keys], rotation=45, ha='right')
        ax.set_yticklabels([str(k) for k in ratio_keys])

        for i in range(n_ratios):
            for j in range(n_ratios):
                val = jacc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)

    for idx in range(n_exits, rows * cols):
        axes[idx // cols][idx % cols].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_exit_correspondence(exit_indices_by_ratio, full_key: str, slim_key: str, out_path: str):
    """Show how samples exiting at each full-model exit split across slim-model exits."""
    full_sets = [set(lst) for lst in exit_indices_by_ratio[full_key]['exit_sample_indices']]
    slim_sets = [set(lst) for lst in exit_indices_by_ratio[slim_key]['exit_sample_indices']]

    n_full = len(full_sets)
    n_slim = len(slim_sets)

    overlap_counts = np.zeros((n_full, n_slim), dtype=float)
    for i in range(n_full):
        for j in range(n_slim):
            overlap_counts[i, j] = len(full_sets[i] & slim_sets[j])

    fig, ax = plt.subplots(figsize=(max(5, n_slim * 1.2), max(4, n_full * 0.8)))
    im = ax.imshow(overlap_counts, aspect='auto', cmap='Blues')
    ax.set_title(f'Exit correspondence: full {full_key} -> slim {slim_key}')
    ax.set_xlabel(f'Slim exits ({slim_key})')
    ax.set_ylabel(f'Full exits ({full_key})')
    ax.set_xticks(list(range(n_slim)))
    ax.set_yticks(list(range(n_full)))

    for i in range(n_full):
        for j in range(n_slim):
            val = overlap_counts[i, j]
            ax.text(j, i, f"{int(val)}", ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _ensure_eval_logits(eval_obj, model):
    """Ensure eval_obj has tester and test logits/targets available."""
    if not hasattr(eval_obj, 'tester') or eval_obj.tester is None:
        eval_obj.model = model
        eval_obj.args.policy = model.config.policy
        eval_obj.n_exits = len(model.config.exits)
        eval_obj.args.n_exits = eval_obj.n_exits
        eval_obj.model.to(eval_obj.device)
        eval_obj.tester = Tester(eval_obj.model, eval_obj.args)

    if not hasattr(eval_obj, 'test_exits_preds') or not hasattr(eval_obj, 'test_targets'):
        eval_obj.test_exits_preds, eval_obj.test_targets, eval_obj.test_all_sample_exits_logits = (
            eval_obj.tester.calc_logtis(eval_obj.test_dataloader)
        )

    if not hasattr(eval_obj, 'valid_exits_preds') or not hasattr(eval_obj, 'valid_targets'):
        eval_obj.valid_exits_preds, eval_obj.valid_targets, eval_obj.valid_all_sample_exits_logits = (
            eval_obj.tester.calc_logtis(eval_obj.valid_dataloader)
        )




if __name__ == '__main__':
    args = args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    eval_dir = args.suffix
    args.img_dir = eval_dir + "/img"
    eval = Eval(args=args)

    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f and '.png' not in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    # print(model_paths)
    for model_path in model_paths:
        if  args.policy in model_path and 'G_' not in model_path and 'loss' not in model_path and 'acc' not in model_path and 'distance' not in model_path and 'budget' not in model_path and 'exit_indices' not in model_path:
            # if 'darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.85-0.7]' in model_path: continue
            if 'darkfl' not in model_path: continue
            # print(model_path)
            eval._log((f'eval model:{os.path.basename(model_path)}').center(120, '='))
            full_model = load_model_eval(args, model_path+'.pth', config_path=model_path+'.json')
            slim_ratios = full_model.config.slim_ratios if full_model.config.slimmable else [1.0]
            # print(slim_ratios)
            slim_x_list = {}
            slim_y_list = {}
            exit_indices_by_ratio = {}
            probs = None
            # eval loop for each slim ratio
            for ratio in slim_ratios:
                # print(f"Evaluating at slim ratio: {ratio}")
                if full_model.config.slimmable:
                    from utils.modelload.slimmable import set_width_ratio
                    
                    eval._log(f'Setting width ratio to {ratio}')
                    set_width_ratio(ratio, full_model)
                    eval.eval(model_path+'.pth', model_path+'.json', model=full_model)

                    eval_path = eval.eval_json_path
                    dct = json.loads(open(eval_path, 'r').read())
                    x = dct['flops']
                    y = dct['test']
                    slim_x_list[ratio] = x
                    slim_y_list[ratio] = y

                    eval.tester = Tester(full_model, eval.args)
                    valid_exits_preds, valid_targets, _ = eval.tester.calc_logtis(eval.valid_dataloader)
                    test_exits_preds, test_targets, _ = eval.tester.calc_logtis(eval.test_dataloader)

                    if probs is None:
                        rnd = 40
                        p_int = 20
                        _p = torch.tensor([p_int * (1.0 / (rnd / 2))], dtype=torch.float32).to(eval.tester.device)
                        probs = torch.exp(
                            torch.log(_p) * torch.tensor([(i + 1) * 4 for i in range(eval.tester.n_exits)]).to(eval.tester.device)
                        )
                        probs /= probs.sum()

                    flops = getattr(eval.tester, 'flops', [float(i + 1) for i in range(eval.tester.n_exits)])
                    _acc_val, _exp_val, T = eval.tester.dynamic_eval_find_threshold(
                        valid_exits_preds, valid_targets, probs, flops
                    )
                    acc, exp_flops, exit_sample_indices = eval.tester.dynamic_eval_with_threshold(
                        test_exits_preds, test_targets, flops, T, return_exit_indices=True
                    )
                    exit_indices_by_ratio[str(ratio)] = {
                        'acc': float(acc),
                        'expected_flops': float(exp_flops),
                        'T': [float(v) for v in T.detach().cpu().tolist()],
                        'exit_sample_indices': exit_sample_indices,
                        'exit_counts': [len(lst) for lst in exit_sample_indices],
                    }
                    # print(exit_indices_by_ratio[str(ratio)]['exit_counts'])
            
            fx, fy = fuse_curves_take_max(slim_x_list, slim_y_list, ref_ratio=1.0)
            area, acc = area_under_fitted_curve(fy, fx)
            print(f"AUC: {area}, ALL SLIM Budgeted Acc: {acc}")
            ratio_1_eval_path = eval.eval_dir+eval.model_path+f'_slim_1.0_exits_[2, 5, 8, 11]_eval.json'
            with open(ratio_1_eval_path, 'r') as f:
                ratio_1_dct = json.load(f)
            merged = {'all_slim_budgeted_acc': acc, **{k: v for k, v in ratio_1_dct.items()}}
            with open(ratio_1_eval_path, 'w') as f:
                json.dump(merged, f)

            if exit_indices_by_ratio:
                # print('in')
                os.makedirs(args.img_dir, exist_ok=True)
                jacc_path = os.path.join(args.img_dir, f"{eval.model_path}_slim_exit_jaccard.png")
                _plot_jaccard_matrices(exit_indices_by_ratio, jacc_path)
                print(f"Saved exit jaccard matrices to: {jacc_path}")

                ratio_items = []
                for k in exit_indices_by_ratio.keys():
                    try:
                        ratio_items.append((float(k), k))
                    except Exception:
                        ratio_items.append((float('inf'), k))
                ratio_items.sort(key=lambda t: t[0])
                ratio_keys = [k for _v, k in ratio_items]

                if len(ratio_keys) >= 2:
                    full_key = ratio_keys[-1]
                    for slim_key in ratio_keys:
                        if slim_key == full_key:
                            continue
                        corr_path = os.path.join(
                            args.img_dir, f"{eval.model_path}_exit_correspondence_{full_key}_vs_{slim_key}.png"
                        )
                        _plot_exit_correspondence(exit_indices_by_ratio, full_key, slim_key, corr_path)
                        print(f"Saved exit correspondence to: {corr_path}")
            