import os
import torch
import torch.nn as nn
import math
import importlib

from tqdm import tqdm

from utils.options import args_parser
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval



class Eval():
    def __init__(self, args):
        self.args = args
        self.if_mode = args.if_mode
        self.device = args.device
        args.valid_ratio = 0.2
        self.valid_dataset, self.valid_dataloader = load_dataset_loader(args=args, eval_valids=True)
        self.test_dataset, self.test_dataloader = load_dataset_loader(args=args, need_process=True)
        self.eval_output_path = f'./{args.suffix}/eval.txt'
        self.eval_output = open(self.eval_output_path, 'a')
        
        
    def eval(self, model_path, config_path):
        self.eval_output.write(((f'eval model:{os.path.basename(model_path)}').center(80, '=')+'\n'))
        self.model = load_model_eval(self.args, model_path, config_path)
        self.n_exits = len(self.model.config.exits)
        self.args.n_exits = self.n_exits
        self.model.to(self.device)
        self.tester = Tester(self.model, self.args)
        
        self.test_exits_preds, self.test_targets = self.tester.calc_logtis(self.test_dataloader)
        self.valid_exits_preds, self.valid_targets = self.tester.calc_logtis(self.valid_dataloader)
        
        self.eval_output.write('logits calc compeleted\n')
        
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
        self.eval_output.write('Anytime:\n{}, avg:{}\n'.format(acc_list, sum(acc_list) / len(acc_list)))
        self.eval_output.flush()
        
    
    def budgeted(self,):
        rnd = 40
        # TODO flops need to be measured
        flops = [i+1 for i in range(self.n_exits)]
        for p in range(1, rnd):
            # self.eval_output.write("\n*********************\n")
            _p = torch.tensor([p * (1.0/(rnd/2))], dtype=torch.float32).to(self.device)
            probs = torch.exp(torch.log(_p) * torch.tensor([i+1 for i in range(self.n_exits)]).to(self.device))
            probs /= probs.sum()
            
            acc_val, _, T = self.tester.dynamic_eval_find_threshold(self.valid_exits_preds, self.valid_targets, probs, flops)
            acc_test, exp_flops = self.tester.dynamic_eval_with_threshold(self.test_exits_preds, self.test_targets, flops, T)
            self.eval_output.write('p: {:d}, valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}\n'.format(p, acc_val, acc_test, exp_flops))
            # self.eval_output.write('{} {} {}\n'.format(p, exp_flops.item(), acc_test))
            self.eval_output.flush()
            
            
class Tester(object):
    def __init__(self, model, args):
        self.args = args
        self.device = self.args.device
        self.model = model
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        self.n_exits = args.n_exits
        
        args.exits_num = self.n_exits
        args.policy = self.model.config.policy
        policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
        self.policy = policy_module.Policy(args)
    
    def calc_logtis(self, dataloader):
        self.model.eval()
        all_sample_exits_logits = [[] for _ in range(self.n_exits)]
        all_sample_targets = []
        for i, data in enumerate(dataloader):
            batch = {}
            for key in data.keys():
               batch[key] = data[key].to(self.device)
            y = batch['labels'].view(-1)
            all_sample_targets.append(y)
            with torch.no_grad():
                exits_logits = self.policy(self.model(**batch))
                for i, exit_logits in enumerate(exits_logits):
                    _t = self.softmax(exit_logits)
                    all_sample_exits_logits[i].append(_t)
        
        for i in range(self.n_exits):
            all_sample_exits_logits[i] = torch.cat(all_sample_exits_logits[i], dim=0)
        
        size = (len(all_sample_exits_logits), all_sample_exits_logits[0].size(0), all_sample_exits_logits[0].size(1))
        ts_preds = torch.zeros(size=size).to(self.device)
        for i in range(len(all_sample_exits_logits)):
            ts_preds[i] = all_sample_exits_logits[i]
            
        all_sample_targets = torch.cat(all_sample_targets, dim=0)
        return ts_preds, all_sample_targets
    
    def dynamic_eval_find_threshold(self, preds, targets, p, flops):
        """
            preds: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_exits, n_sample, n_class = preds.size()
        max_preds, argmax_preds = preds.max(dim=2, keepdim=False)
        _, sorted_idx = max_preds.sort(dim=1, descending=True)
        filtered = torch.zeros(n_sample)
        T = torch.tensor([1e8 for _ in range(n_exits)]).to(self.device)
        
        for k in range(n_exits - 1):
            crt, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    filtered[ori_idx] = 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
        
        T[n_exits-1] = -1e8
        
        # crt_rec 在各个出口的准确率，exp 从各个出口出来的sample数量
        crt_rec, exp = torch.zeros(n_exits), torch.zeros(n_exits)
        crt, expected_flops = 0, 0
        for i in range(n_sample):
            glod_label = targets[i]
            for k in range(n_exits):
                if max_preds[k][i].item() >= T[k]:
                    if (int(glod_label.item())) == int(argmax_preds[k][i].item()):
                        crt += 1
                        crt_rec[k] += 1
                    exp[k] += 1
                    break
        
        for k in range(n_exits):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]

        return crt * 100 / n_sample, expected_flops, T
        
    def dynamic_eval_with_threshold(self, preds, targets, flops, T):
        n_exits, n_sample, _ = preds.size()
        max_preds, argmax_preds = preds.max(dim=2, keepdim=False)
        
        # crt_rec 在各个出口的正确个数，exp 从各个出口出来的sample数量
        crt_rec, exp = torch.zeros(n_exits), torch.zeros(n_exits)
        crt, expected_flops = 0, 0
        for i in range(n_sample):
            glod_label = targets[i]
            for k in range(n_exits):
                if max_preds[k][i].item() >= T[k]:
                    if (int(glod_label.item())) == int(argmax_preds[k][i].item()):
                        crt += 1
                        crt_rec[k] += 1
                    exp[k] += 1
                    break
        
        for k in range(n_exits):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
        
        return crt * 100 / n_sample, expected_flops
        
    
if __name__ == '__main__':
    args = args_parser()
    eval = Eval(args=args)
    eval_dir = args.suffix
    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    for model_path in model_paths:
        if 'ours2' in model_path and 'base' in model_path and 'False' in model_path:
            eval.eval(model_path+'.pth', model_path+'.json')