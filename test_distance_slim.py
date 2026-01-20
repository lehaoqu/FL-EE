import numpy as np
import random
import torch
import os
import json
import math
import matplotlib.pyplot as plt

from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval
from eval import Tester



class T:
    def __init__(self):
        seed = 1117
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        class A: pass
        self.args = A()
        self.args.model = 'vit'
        self.args.dataset = 'cifar100_noniid1000'
        self.args.policy = 'boosted'
        self.args.alg = 'depthfl'
        self.args.blocks = (2,5,8,11)
        self.args.load_path = ''
        self.args.ft = 'full'
        self.args.device = 2
        self.args.valid_ratio = 0.2
        self.args.total_num = 100
        self.args.bs = 32
        self.args.ensemble_weight = 0.2
        self.args.config_path = 'EXPS2/BASE_CIFAR_ALL/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.8].json'
        self.args.model_path = 'EXPS2/BASE_CIFAR_ALL/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.8].pth'

        self.args.origin_config_path = 'EXPS2/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
        self.args.origin_model_path = 'EXPS2/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'

    def load_model(self):
        self.model = load_model_eval(self.args, model_path=self.args.model_path, config_path=self.args.config_path)
        self.model.eval()
        self.model.to(self.args.device)

        self.origin_model = load_model_eval(self.args, model_path=self.args.origin_model_path, config_path=self.args.origin_config_path)
        self.origin_model.eval()
        self.origin_model.to(self.args.device)
    

    def load_data(self):
        self.valid_dataset, self.valid_dataloader = load_dataset_loader(args=self.args, eval_valids=True, shuffle=False)
        self.test_dataset, self.test_dataloader = load_dataset_loader(args=self.args, file_name='test', shuffle=False)


    def test_distance_slim(self):
        self.load_model()
        self.load_data()
        slim_ratios = self.model.config.slim_ratios if self.model.config.slimmable else [1.0]

        # Ensure device is a valid torch device string.
        if isinstance(self.args.device, int):
            self.args.device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"
        elif isinstance(self.args.device, str) and self.args.device.isdigit():
            self.args.device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"

        self.args.n_exits = len(self.model.config.exits)
        self.args.policy = self.model.config.policy

        tester = Tester(self.model, self.args, measure_flops=False)
        # origin_tester = Tester(self.origin_model, self.args, measure_flops=False)

        # Use the same p->probs construction as Eval.budgeted, pick p=20 (i.e. _p=1.0 when rnd=40)
        rnd = 40
        p_int = 20
        _p = torch.tensor([p_int * (1.0 / (rnd / 2))], dtype=torch.float32).to(tester.device)
        probs = torch.exp(torch.log(_p) * torch.tensor([(i + 1) * 4 for i in range(tester.n_exits)]).to(tester.device))
        probs /= probs.sum()

        exit_indices_by_ratio = {}

        if self.model.config.slimmable:
            from utils.modelload.slimmable import set_width_ratio

        slim_ratios = slim_ratios + [2.0]
        for ratio in slim_ratios:
            if ratio != 2.0:
                if self.model.config.slimmable:
                    set_width_ratio(ratio, self.model)
                tester.model = self.model
            else:
                # Use the origin full model
                tester.model = self.origin_model

            test_preds, test_targets, _ = tester.calc_logtis(self.test_dataloader)

            # flops is not needed for indices recording; pass a placeholder.
            flops_placeholder = [float(i + 1) for i in range(tester.n_exits)]
            acc, exp_flops, T, exit_sample_indices = tester.dynamic_eval_find_threshold(
                test_preds, test_targets, probs, flops_placeholder, return_exit_indices=True
            )

            exit_indices_by_ratio[str(ratio)] = {
                'acc': float(acc),
                'expected_flops': float(exp_flops),
                'T': [float(v) for v in T.detach().cpu().tolist()],
                'exit_sample_indices': exit_sample_indices,
                'exit_counts': [len(lst) for lst in exit_sample_indices],
            }
            print(f"Slim ratio: {ratio}, Acc: {acc}, exit counts: {[len(lst) for lst in exit_sample_indices]}")

        out_path = os.path.splitext(self.args.model_path)[0] + "_exit_indices.json"
        with open(out_path, 'w') as f:
            json.dump(exit_indices_by_ratio, f)
        print(f"Saved exit indices to: {out_path}")

        self._visualize_exit_indices(exit_indices_by_ratio, out_path)
        


    def _visualize_exit_indices(self, exit_indices_by_ratio, out_json_path: str):
        if not exit_indices_by_ratio:
            print("No exit indices to visualize.")
            return

        # Sort ratios numerically when possible
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

        # Build set cache for Jaccard computations
        exit_sets = []
        for rk in ratio_keys:
            sets_for_ratio = [set(lst) for lst in exit_indices_by_ratio[rk]['exit_sample_indices']]
            exit_sets.append(sets_for_ratio)

        # 1) Heatmap: counts per ratio/exit
        counts = np.zeros((n_ratios, n_exits), dtype=float)
        for i, rk in enumerate(ratio_keys):
            counts[i, :] = exit_indices_by_ratio[rk]['exit_counts']

        fig1, ax1 = plt.subplots(figsize=(max(6, n_exits * 1.2), max(4, n_ratios * 0.6)))
        im1 = ax1.imshow(counts, aspect='auto', cmap='viridis')
        ax1.set_title('Exit sample counts by slim ratio')
        ax1.set_xlabel('Exit')
        ax1.set_ylabel('Slim ratio')
        ax1.set_xticks(list(range(n_exits)))
        ax1.set_yticks(list(range(n_ratios)))
        ax1.set_yticklabels([str(k) for k in ratio_keys])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        fig1.tight_layout()

        counts_path = os.path.splitext(out_json_path)[0] + "_exit_counts.png"
        fig1.savefig(counts_path, dpi=200)
        plt.close(fig1)
        print(f"Saved counts heatmap to: {counts_path}")

        # 2) Jaccard similarity matrices per exit
        cols = min(3, n_exits)
        rows = int(math.ceil(n_exits / cols))
        fig2, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

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
            im = ax.imshow(jacc, vmin=0.0, vmax=1.0, cmap='magma')
            ax.set_title(f'Exit {e}: Jaccard')
            ax.set_xticks(list(range(n_ratios)))
            ax.set_yticks(list(range(n_ratios)))
            ax.set_xticklabels([str(k) for k in ratio_keys], rotation=45, ha='right')
            ax.set_yticklabels([str(k) for k in ratio_keys])

            # Annotate each cell with the similarity value
            for i in range(n_ratios):
                for j in range(n_ratios):
                    val = jacc[i, j]
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=8)

        # Hide empty subplots if any
        for idx in range(n_exits, rows * cols):
            axes[idx // cols][idx % cols].axis('off')

        fig2.tight_layout()
        jacc_path = os.path.splitext(out_json_path)[0] + "_exit_jaccard.png"
        fig2.savefig(jacc_path, dpi=200)
        plt.close(fig2)
        print(f"Saved Jaccard heatmaps to: {jacc_path}")

        # 3) Cross-model exit correspondence (full vs slim)
        if len(ratio_keys) >= 2:
            full_key = ratio_keys[-1]
            for slim_key in ratio_keys[:-1]:
                self._visualize_exit_correspondence(exit_indices_by_ratio, full_key, slim_key, out_json_path)

    def _visualize_exit_correspondence(self, exit_indices_by_ratio, full_key: str, slim_key: str, out_json_path: str):
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

        corr_path = os.path.splitext(out_json_path)[0] + f"_exit_correspondence_{full_key}_vs_{slim_key}.png"
        fig.savefig(corr_path, dpi=200)
        plt.close(fig)
        print(f"Saved exit correspondence to: {corr_path}")


if __name__ == '__main__':
    T().test_distance_slim()
        
