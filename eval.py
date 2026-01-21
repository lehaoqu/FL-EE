import os
import torch
import torch.nn as nn
import math
import importlib
from PIL import Image
import numpy as np
import argparse
import json
import tempfile
import random

from tqdm import tqdm
from transformers import BertTokenizer
import utils.modelload.slimmable as slimmable_module
from utils.train_utils import get_flops

from utils.options import args_parser
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset
from dataset.imagenet_dataset import TinyImageNetClassificationDataset
from dataset.speechcmd_dataset import SPEEDCMDSClassificationDataset



class Eval():
    def __init__(self, args):
        self.args = args
        self.if_mode = args.if_mode
        self.device = args.device
        args.valid_ratio = 0.2
        self.valid_dataset, self.valid_dataloader = load_dataset_loader(args=args, eval_valids=True, shuffle=False)
        self.test_dataset, self.test_dataloader = load_dataset_loader(args=args, file_name='test', shuffle=False)
        self.eval_output_path = f'./{args.suffix}/eval.txt'
        self.eval_output = open(self.eval_output_path, 'w')
        self.eval_dir = f'./{args.suffix}/'
        self.img_dir = args.img_dir


    def _log(self, message: str):
        """Write to eval file and echo to terminal."""
        msg = message if message.endswith('\n') else message + '\n'
        self.eval_output.write(msg)
        self.eval_output.flush()
        print(message, flush=True)


    def _atomic_json_dump(self, path: str, data) -> None:
        """Atomically write JSON to avoid empty/corrupt files on interruption."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        
        
    def eval(self, model_path, config_path, model=None):

        if 'cifar' in self.args.dataset:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        
        base_name = os.path.basename(model_path)
        name_without_extension = os.path.splitext(base_name)[0]
        self.model_path = name_without_extension
        
        if model is None:
            self.model = load_model_eval(self.args, model_path, config_path)
        else:
            self.model = model
        
        self.args.policy = self.model.config.policy
        self.n_exits = len(self.model.config.exits)
        self.args.n_exits = self.n_exits
        self.model.to(self.device)
        self.tester = Tester(self.model, self.args)

        # parser = argparse.ArgumentParser()
        # policy_module = importlib.import_module(f'trainer.policy.{self.model.config.policy}')
        # policy_parser = parser.add_subparsers(dest='policy_parser')
        # policy_parser = policy_module.add_args(policy_parser)
        # policy_args = policy_parser.parse_args()
        
        # for arg in vars(policy_args):
        #     print(arg, getattr(policy_args, arg))
        #     setattr(self.args, arg, getattr(policy_args, arg))
            
        # for arg in vars(args):
        #     self.eval_output.write(f"{arg}: {getattr(args, arg)}\n")
             
        self.args.policy = self.model.config.policy
        self.n_exits = len(self.model.config.exits)
        self.args.n_exits = self.n_exits
        self.model.to(self.device)
        self.tester = Tester(self.model, self.args)
        
        self.test_exits_preds, self.test_targets, self.test_all_sample_exits_logits = self.tester.calc_logtis(self.test_dataloader)
        self.valid_exits_preds, self.valid_targets, self.valid_all_sample_exits_logits  = self.tester.calc_logtis(self.valid_dataloader)
        
        self._log('logits calc compeleted')
        
        if self.args.cosine is True:
            self.test_cos_exits, self.test_all_sample_cos_exits = self.cos_similiarity(self.test_all_sample_exits_logits)
            # self.valid_cos_exits, self.valid_all_sample_cos_exits = self.cos_similiarity(self.valid_all_sample_exits_logits)
            spec_labels = [8, 13]
            for spec_label in spec_labels:
                self.sort(self.test_all_sample_cos_exits, self.test_dataset, spec_label)
            # self.sort(self.valid_all_sample_cos_exits, self.valid_dataset)
            

        all_exits = self.model.config.exits
        for idx in range(len(all_exits)):
            self._log(f'Evaluating with exits: {all_exits[:idx+1]}')
            cur_exits = all_exits[:idx+1]
            self.model.config.exits = cur_exits
            self.n_exits = len(self.model.config.exits)
            self.args.n_exits = self.n_exits

            if self.model.config.slimmable:
                # print('current width ratio:', slimmable_module.CURRENT_WIDTH_RATIO)
                self.eval_json_path = self.eval_dir+self.model_path+f'_slim_{slimmable_module.CURRENT_WIDTH_RATIO}_exits_{self.model.config.exits}_eval.json'
            else:
                self.eval_json_path = self.eval_dir+self.model_path+f'_exits_{self.model.config.exits}_eval.json'

            if os.path.exists(self.eval_json_path):
                try:
                    with open(self.eval_json_path, 'r') as f:
                        raw = f.read().strip()
                    if not raw:
                        raise ValueError("empty eval json")
                    dct = json.loads(raw)
                except Exception as e:
                    self._log(f"[WARN] Skip broken eval json: {self.eval_json_path} ({e}); will re-evaluate.")
                else:
                    if 'budgeted_acc' not in dct:
                        x = dct.get('flops', [])
                        y = dct.get('test', [])
                        from utils.train_utils import area_under_fitted_curve
                        _area, acc = area_under_fitted_curve(y, x)
                        dct['budgeted_acc'] = acc
                        self._atomic_json_dump(self.eval_json_path, dct)
                    continue
            if self.args.if_mode == 'anytime':
                self.anytime()
            elif self.args.if_mode == 'budgeted':
                self.budgeted()
            else:
                self.budgeted()
                self.anytime()
        
    
    def anytime(self,):
        crt_list = [0 for _ in range(self.n_exits)]
        for i in range(self.n_exits):
            _, predicted = torch.max(self.test_exits_preds[i], 1)
            crt_list[i] += (predicted == self.test_targets).sum().item()
        
        acc_list = [100 * crt_list[i] / self.test_targets.shape[0] for i in range(self.n_exits)]
        self._log('Anytime:\n{}, avg:{}'.format(acc_list, sum(acc_list) / len(acc_list)))

        
    def budgeted(self,):
        rnd = 40
        # TODO flops need to be measured
        # use measured FLOPs from tester when available
        # flops = getattr(self.tester, 'flops', [i+1 for i in range(self.n_exits)])
        flops = self.tester.flops[:self.n_exits]
        self._log('Exits FLOPs: {}'.format(flops))
        acc_test_list = ''
        exp_flops_list = ''
        
        acc_test_np, acc_val_np, exp_flops_np = [],[],[]
        
        for p in range(1, rnd):
            # self.eval_output.write("\n*********************\n")
            _p = torch.tensor([p * (1.0/(rnd/2))], dtype=torch.float32).to(self.device)
            probs = torch.exp(torch.log(_p) * torch.tensor([(i+1)*4 for i in range(self.n_exits)]).to(self.device))
            probs /= probs.sum()
            
            acc_val, _, T = self.tester.dynamic_eval_find_threshold(self.valid_exits_preds[:self.n_exits], self.valid_targets, probs, flops)
            acc_test, exp_flops = self.tester.dynamic_eval_with_threshold(self.test_exits_preds[:self.n_exits], self.test_targets, flops, T)
            acc_test_list += (str(acc_test)+'\n')
            exp_flops_list += (str(exp_flops)+'\n')
            acc_test_np.append(float(acc_test))
            # acc_val_np.append(acc_val)
            exp_flops_np.append(float(exp_flops))
            self._log('p: {:d}, test acc: {:.3f}, test flops: {:.2f}'.format(p, float(acc_test), float(exp_flops)))
            # self._log('{} {} {}'.format(p, exp_flops.item(), acc_test))
        # self.eval_output.write(acc_test_list)
        # self.eval_output.write(exp_flops_list)
        # if self.model.config.slimmable:
        #     self.file_path = self.eval_json+self.model_path+f'_slim_{slimmable_module.CURRENT_WIDTH_RATIO}_eval.json'
        # else:
        #     self.file_path = self.eval_json+self.model_path+'_eval.json'
        y = acc_test_np
        x = exp_flops_np
        from utils.train_utils import area_under_fitted_curve
        area, acc = area_under_fitted_curve(y, x)
        self._log(f'Budgeted AUC: {area}, AVG ACC: {acc}')
        self._atomic_json_dump(self.eval_json_path, {'budgeted_acc': acc, 'test':acc_test_np, 'val':acc_val_np, 'flops':exp_flops_np})
            
    def cos_similiarity(self, all_sample_exits_logits):
        sample_num = all_sample_exits_logits[0].size(0)
        all_sample_cos_exits = [[] for _ in range(sample_num)]
        
        for i in range(sample_num):
            last_logits = all_sample_exits_logits[-1][i].unsqueeze(0)
            for exit_idx in range(self.n_exits):
                exit_logits = all_sample_exits_logits[exit_idx][i].unsqueeze(0)
                cos_similar = nn.functional.cosine_similarity(exit_logits, last_logits, dim=1).cpu().item()
                all_sample_cos_exits[i].append(cos_similar)
        print(len(all_sample_cos_exits), len(all_sample_cos_exits[0]))
        cos_exits_array = np.array(all_sample_cos_exits).transpose()
        cos_exits_means = np.mean(cos_exits_array, axis=1)
        print(cos_exits_means)
        self._log(str(cos_exits_means))
        return cos_exits_means, all_sample_cos_exits
    
    
    def sort(self, all_sample_cos_exits, dataset, spec_label):
        
        if 'cifar' in self.args.dataset:
        
            label_list = []
            for i in range(len(dataset)):
                if dataset[i]['labels'].cpu().item() == spec_label:
                    label_list.append(i)
            dataset_label = [{'labels':dataset[i]['labels'], 'pixel_values':dataset[i]['pixel_values']} for i in label_list]
            
            label_sample_cos_exits = [all_sample_cos_exits[i] for i in label_list]
            all_sample_score = [sum(cos_exits) for i, cos_exits in enumerate(label_sample_cos_exits)]
            score_array = np.array(all_sample_score)
            indices = np.argsort(-score_array)
            
            n = 5
            div_points = np.linspace(0, len(all_sample_score)-1, n).astype(np.uint).tolist()
            
            for dlevel, level_idx in enumerate(div_points):
                label  = dataset_label[indices[level_idx]]['labels']
                sample = dataset_label[indices[level_idx]]['pixel_values']

                sample = sample.numpy().reshape(3,32,32) if isinstance(sample, torch.Tensor) else sample.reshape(3,32,32)
                array = np.transpose(sample, (1, 2, 0))
                img = Image.fromarray(array.astype(np.uint8))
                img.save(f'{self.img_dir}/{self.model_path}_dlevel_{dlevel}_l_{label}.png')
                self._log(f'{self.model_path}_dlevel_{dlevel}_l_{label}: {all_sample_cos_exits[indices[level_idx]]}')
        else:
        
            label_list = []
            for i in range(len(dataset)):
                if dataset[i]['labels'].cpu().item() == spec_label:
                    label_list.append(i)
            dataset_label = [{'labels':dataset[i]['labels'], 'input_ids':dataset[i]['input_ids']} for i in label_list]
            
            label_sample_cos_exits = [all_sample_cos_exits[i] for i in label_list]
            all_sample_score = [sum(cos_exits) for i, cos_exits in enumerate(label_sample_cos_exits)]
            score_array = np.array(all_sample_score)
            indices = np.argsort(-score_array)
            
            n = 5
            div_points = np.linspace(0, len(all_sample_score)-1, n).astype(np.uint).tolist()
            
            for dlevel, level_idx in enumerate(div_points):
                label = dataset_label[indices[level_idx]]['labels'].item()
                sample = dataset_label[indices[level_idx]]['input_ids']

                tokenizer = BertTokenizer.from_pretrained('./models/bert_uncased_L-12_H-128_A-2')
                detokenized_tokens = tokenizer.convert_ids_to_tokens(sample)
                filtered_tokens = [token for token in detokenized_tokens if token not in ("[CLS]", "[PAD]")]
                detokenized_text = " ".join(filtered_tokens)
                self._log(f'{self.model_path}_dlevel_{dlevel}_l_{label}: {all_sample_cos_exits[indices[level_idx]]}')
                self._log(f'label: {label}, sample: {detokenized_text}')
    
            
            
class Tester(object):
    def __init__(self, model, args, *, measure_flops: bool = True):
        self.args = args
        self.device = self.args.device
        self.model = model
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        self.n_exits = args.n_exits
        
        args.exits_num = self.n_exits
        args.policy = self.model.config.policy
        policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
        self.policy = policy_module.Policy(args)
        # FLOPs measurement can be expensive; allow disabling for analysis.
        if measure_flops:
            self.flops = []
            for exit_idx in range(self.n_exits):
                exit_flops = get_flops(args, self.model, stop_exit=exit_idx)
                self.flops.append(exit_flops)
        else:
            self.flops = [float(i + 1) for i in range(self.n_exits)]
        # print(self.flops)
        # print('============')
    
    def adapt_batch(self, data):
        batch = {}
        for key in data.keys():
            batch[key] = data[key].to(self.device)
            if key == 'pixel_values':
                if 'cifar' in self.args.dataset:
                    batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
                elif 'imagenet' in self.args.dataset:
                    batch[key] = TinyImageNetClassificationDataset.transform_for_vit(batch[key])
                elif 'speechcmds' in self.args.dataset:
                    batch[key] = SPEEDCMDSClassificationDataset.transform_for_vit(batch[key])
                else:
                    batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
        label = batch['labels'].view(-1)
        return batch, label
    
    
    def calc_logtis(self, dataloader):
        self.model.eval()
        all_sample_exits_logits = [[] for _ in range(self.n_exits)]
        all_sample_targets = []
        for i, data in enumerate(dataloader):           
            batch, y = self.adapt_batch(data)
            
            all_sample_targets.append(y)
            with torch.no_grad():
                exits_logits = self.policy(self.model(**batch))
                assert len(exits_logits) == self.n_exits, f"Expected {self.n_exits} exits, got {len(exits_logits)}"
                for i, exit_logits in enumerate(exits_logits):
                    # _t = self.softmax(exit_logits)
                    # all_sample_exits_logits[i].append(_t)
                    all_sample_exits_logits[i].append(exit_logits)
        
        for i in range(self.n_exits):
            all_sample_exits_logits[i] = torch.cat(all_sample_exits_logits[i], dim=0)
        
        size = (len(all_sample_exits_logits), all_sample_exits_logits[0].size(0), all_sample_exits_logits[0].size(1))
        ts_preds = torch.zeros(size=size).to(self.device)
        for i in range(len(all_sample_exits_logits)):
            ts_preds[i] = all_sample_exits_logits[i]
            
        all_sample_targets = torch.cat(all_sample_targets, dim=0)
        return ts_preds, all_sample_targets, all_sample_exits_logits
    
    def dynamic_eval_find_threshold(self, preds, targets, p, flops, return_exit_indices: bool = False):
        """
            preds: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_exits, n_sample, _n_class = preds.size()
        max_preds, _argmax_preds = preds.max(dim=2, keepdim=False)
        sorted_idx = max_preds.argsort(dim=1, descending=True)

        filtered = torch.zeros(n_sample, device=max_preds.device, dtype=torch.bool)
        T = torch.full((n_exits,), 1e8, device=max_preds.device, dtype=max_preds.dtype)

        # Only loop over exits; selection within each exit is vectorized.
        for k in range(n_exits - 1):
            try:
                pk = float(p[k].item()) if torch.is_tensor(p) else float(p[k])
            except Exception:
                pk = float(p[k])
            out_n = int(math.floor(n_sample * pk))
            if out_n <= 0:
                continue

            idx_sorted = sorted_idx[k]  # (n_sample,)
            unused_mask = ~filtered[idx_sorted]
            unused_cnt = int(unused_mask.sum().item())
            if unused_cnt <= 0:
                continue

            # If out_n >= remaining, mark all remaining as filtered and keep T[k]=1e8
            # to match the original behavior.
            if out_n >= unused_cnt:
                selected = idx_sorted[unused_mask]
                filtered[selected] = True
                continue

            selected = idx_sorted[unused_mask][:out_n]
            filtered[selected] = True
            T[k] = max_preds[k, selected[-1]]

        T[n_exits - 1] = -1e8

        if return_exit_indices:
            acc, expected_flops, exit_sample_indices = self.dynamic_eval_with_threshold(
                preds, targets, flops, T, return_exit_indices=True
            )
            return acc, expected_flops, T, exit_sample_indices

        acc, expected_flops = self.dynamic_eval_with_threshold(preds, targets, flops, T)
        return acc, expected_flops, T
        
    def dynamic_eval_with_threshold(self, preds, targets, flops, T, return_exit_indices: bool = False):
        n_exits, n_sample, _ = preds.size()
        max_preds, argmax_preds = preds.max(dim=2, keepdim=False)

        T = T.to(device=max_preds.device, dtype=max_preds.dtype)
        # mask[k, i] indicates sample i can exit at k
        mask = max_preds >= T.view(-1, 1)
        # Ensure every sample exits somewhere (safety net for extreme logits)
        mask[-1, :] = True

        exit_idx = mask.to(torch.float32).argmax(dim=0)  # (n_sample,)
        sample_idx = torch.arange(n_sample, device=max_preds.device)

        pred_label = argmax_preds[exit_idx, sample_idx]
        correct = pred_label.eq(targets).to(torch.float32)

        exp = torch.bincount(exit_idx, minlength=n_exits).to(torch.float32)

        flops_t = torch.as_tensor(flops, device=max_preds.device, dtype=torch.float32)
        # print(exp.size(), flops_t.size())
        expected_flops = (exp / float(n_sample) * flops_t).sum()

        acc = correct.mean() * 100.0
        acc_f = float(acc.detach().cpu().item())
        exp_f = float(expected_flops.detach().cpu().item())

        if return_exit_indices:
            exit_idx_cpu = exit_idx.detach().cpu()
            exit_sample_indices = [
                (exit_idx_cpu == k).nonzero(as_tuple=False).view(-1).tolist() for k in range(n_exits)
            ]
            return acc_f, exp_f, exit_sample_indices

        return acc_f, exp_f
        
    
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
        if  args.policy in model_path and 'G_' not in model_path and 'loss' not in model_path and 'acc' not in model_path and 'distance' not in model_path and 'budget' not in model_path:
            # print(model_path)
            eval._log((f'eval model:{os.path.basename(model_path)}').center(80, '='))
            eval.eval(model_path+'.pth', model_path+'.json')