import os
import torch
import torch.nn as nn
import math
import importlib
from PIL import Image
import numpy as np
import argparse

from tqdm import tqdm
from transformers import BertTokenizer

from utils.options import args_parser
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset



class Eval():
    def __init__(self, args):
        self.args = args
        self.if_mode = args.if_mode
        self.device = args.device
        args.valid_ratio = 0.2
        self.valid_dataset, self.valid_dataloader = load_dataset_loader(args=args, eval_valids=True, shuffle=False)
        self.test_dataset, self.test_dataloader = load_dataset_loader(args=args, file_name='test', shuffle=False)
        self.eval_output_path = f'./{args.suffix}/eval.txt'
        self.eval_output = open(self.eval_output_path, 'a')
        self.img_dir = args.img_dir
  
        
        
    def eval(self, model_path, config_path):
        if 'cifar' in args.dataset:
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)
        
        base_name = os.path.basename(model_path)
        name_without_extension = os.path.splitext(base_name)[0]
        self.model_path = name_without_extension
        
        
        self.eval_output.write(((f'eval model:{os.path.basename(model_path)}').center(80, '=')+'\n'))
        self.model = load_model_eval(self.args, model_path, config_path)

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
        
        self.eval_output.write('logits calc compeleted\n')
        
        if self.args.cosine is True:
            self.test_cos_exits, self.test_all_sample_cos_exits = self.cos_similiarity(self.test_all_sample_exits_logits)
            # self.valid_cos_exits, self.valid_all_sample_cos_exits = self.cos_similiarity(self.valid_all_sample_exits_logits)
            spec_labels = [8, 13]
            for spec_label in spec_labels:
                self.sort(self.test_all_sample_cos_exits, self.test_dataset, spec_label)
            # self.sort(self.valid_all_sample_cos_exits, self.valid_dataset)
            
        
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
        rnd = 20
        # TODO flops need to be measured
        flops = [i+1 for i in range(self.n_exits)]
        acc_test_list = ''
        exp_flops_list = ''
        for p in range(1, rnd):
            # self.eval_output.write("\n*********************\n")
            _p = torch.tensor([p * (1.0/(rnd/2))], dtype=torch.float32).to(self.device)
            probs = torch.exp(torch.log(_p) * torch.tensor([i+1 for i in range(self.n_exits)]).to(self.device))
            probs /= probs.sum()
            
            acc_val, _, T = self.tester.dynamic_eval_find_threshold(self.valid_exits_preds, self.valid_targets, probs, flops)
            acc_test, exp_flops = self.tester.dynamic_eval_with_threshold(self.test_exits_preds, self.test_targets, flops, T)
            acc_test_list += (str(acc_test)+'\n')
            exp_flops_list += (str(exp_flops.cpu().item())+'\n')
            self.eval_output.write('p: {:d}, valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}\n'.format(p, acc_val, acc_test, exp_flops))
            # self.eval_output.write('{} {} {}\n'.format(p, exp_flops.item(), acc_test))
            self.eval_output.flush()
        self.eval_output.write(acc_test_list)
        self.eval_output.write(exp_flops_list)
        self.eval_output.flush()
            
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
        self.eval_output.write(str(cos_exits_means)+"\n")
        self.eval_output.flush()
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
                self.eval_output.write(f'{self.model_path}_dlevel_{dlevel}_l_{label}: {all_sample_cos_exits[indices[level_idx]]}' + "\n")
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

                tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-12-128-uncased')
                detokenized_tokens = tokenizer.convert_ids_to_tokens(sample)
                filtered_tokens = [token for token in detokenized_tokens if token not in ("[CLS]", "[PAD]")]
                detokenized_text = " ".join(filtered_tokens)
                self.eval_output.write(f'{self.model_path}_dlevel_{dlevel}_l_{label}: {all_sample_cos_exits[indices[level_idx]]}' + "\n")
                self.eval_output.write(f'label: {label}, sample: {detokenized_text}' + "\n")
    
            
            
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
    
    def adapt_batch(self, data):
        batch = {}
        for key in data.keys():
            batch[key] = data[key].to(self.device)
            if key == 'pixel_values':
                if 'cifar' in self.args.dataset:
                    batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
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
    eval_dir = args.suffix
    args.img_dir = eval_dir + "/img"
    eval = Eval(args=args)

    file_names = os.listdir(eval_dir)
    model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f]))
    model_paths = [f'./{eval_dir}/{model_name}' for model_name in model_names]
    for model_path in model_paths:
        print(model_path)
        if 'eefl' in model_path and args.policy in model_path:
            eval.eval(model_path+'.pth', model_path+'.json')