import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import os
import copy
import warnings
warnings.simplefilter('always', UserWarning)

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy, difficulty_measure
from utils.modelload.model import BaseModule
from torch.utils.data import ConcatDataset
from trainer.policy.l2w import MLP_tanh


def add_args(parser):
    parser.add_argument('--is_latent',              default=True, type=bool)
    
    parser.add_argument('--s_epoches',              default=2, type=int)
    parser.add_argument('--s_bs',                   default=32, type=int)
    parser.add_argument('--adaptive_epoches',       action='store_true')
    
    parser.add_argument('--kd_skip',                default=1, type=int)
    parser.add_argument('--kd_begin',               default=0, type=int)
    parser.add_argument('--kd_lr',                  default=5e-2, type=float)
    parser.add_argument('--kd_response_ratio',      default=3, type=float)
    parser.add_argument('--kd_dist_ratio',          default=5, type=float)
    parser.add_argument('--kd_angle_ratio',         default=10, type=float)
    parser.add_argument('--kd_dark_ratio',          default=0, type=float)
    parser.add_argument('--kd_n_iters',             default=5, type=int)
    parser.add_argument('--kd_gap',                 default=1, type=float)
    
    parser.add_argument('--g_skip',                 default=1, type=int)
    parser.add_argument('--g_begin',                default=0, type=int)
    parser.add_argument('--g_lr',                   default=1e-2, type=float)
    parser.add_argument('--g_y',                    default=1, type=float)
    parser.add_argument('--g_div',                  default=1, type=float)
    parser.add_argument('--g_gap',                  default=0, type=float)
    parser.add_argument('--g_diff',                 default=1, type=float)
    parser.add_argument('--g_n_iters',              default=1, type=int)

    parser.add_argument('--kd_direction',           default='sl', type=str)
    parser.add_argument('--kd_join',                default='last', type=str, help='last: only last exit of teacher can teach student model\'s exits')
    parser.add_argument('--agg',                    default='after', type=str)
    
    parser.add_argument('--loss_type',              default='kd', type=str)
    parser.add_argument('--dm',                     default='loss', type=str)
    parser.add_argument('--diff_client_gap',        default=1, type=int)
    parser.add_argument('--diff_generator',         action='store_false')
    
    parser.add_argument('--sw',                     default='learn', type=str, help='how to get weight for students [learn | distance]')
    parser.add_argument('--sw_type',                default='soft', type=str, help='weight [soft | hard]')
    
    parser.add_argument('--exit_p',                 default=30, type=int, help='p of exit policy')
    parser.add_argument('--s_gamma',                default=1, type=float, help='decay of server lr')
    
    return parser




class Client(BaseClient):
    
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        super().__init__(id, args, dataset, model, depth, exits)
        self.diff_distribute, self.sample_exits_diff = None, None
        self.client_crt_rnd = 0
        self.batch_num = len(self.loader_train)
        self.args.diff_client_gap = self.args.diff_client_gap if self.args.diff_generator else 100
        
    
    def train(self):
        self.sample_exits_diff = torch.zeros(len(self.dataset_train), 1).to(self.device)
        self.sample_y = torch.zeros(len(self.dataset_train), 1, dtype=torch.long).to(self.device)
        self.sample_sl = torch.zeros(len(self.dataset_train), 1, dtype=torch.long).to(self.device)
        self.diff_distribute = [1 for _ in range(10)]
        sample_idx = 0
        # eval diff distribution
        # if self.client_crt_rnd % self.args.diff_client_gap == 0:    
        #     for idx, data in enumerate(self.loader_train):
        #         batch, label = self.adapt_batch(data)
        #         with torch.no_grad():
        #             dm_exits_logits, dm_exits_feature = self.server.dm(**batch, rt_feature=True)
        #             dm_exits_logits = self.server.dm_policy(dm_exits_logits)
        #             for index in range(label.shape[0]):
        #                 diff, exits_diff = difficulty_measure([dm_exits_logits[i][index] for i in range(len(dm_exits_logits))], label[index], metric=self.args.dm, rt_exits_diff=True)
        #                 self.sample_exits_diff[sample_idx] = exits_diff
        #                 self.sample_y[sample_idx] = label[index]
        #                 self.diff_distribute[int(diff.cpu().item())] += 1
        #                 if 'attention_mask' in data.keys():
        #                     attention_mask = data['attention_mask'].cpu().tolist()
        #                     sentence_len = len([x for x in attention_mask[index] if x != 0]) -1
        #                     self.sample_sl[sample_idx] = torch.tensor(sentence_len, dtype=torch.long)
        #                 sample_idx += 1

        
        # === train ===
        self.model.to(self.device)
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()

                batch, label = self.adapt_batch(data)
                
                if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                    self.policy.train_meta(self.model, batch, label, self.optim)

                exits_ce_loss, exits_logits = self.policy.train(self.model, batch, label)
                ce_loss = sum(exits_ce_loss)
                if epoch == self.epoch-1:
                    for index in range(label.shape[0]):
                        diff, exits_diff = difficulty_measure([exits_logits[0][index]], label[index], metric=self.args.dm, rt_exits_diff=True)
                        self.sample_exits_diff[sample_idx] = exits_diff.detach()
                        self.sample_y[sample_idx] = label[index]
                        # self.diff_distribute[int(diff.cpu().item())] += 1
                        if 'attention_mask' in data.keys():
                            attention_mask = data['attention_mask'].cpu().tolist()
                            sentence_len = len([x for x in attention_mask[index] if x != 0]) -1
                            self.sample_sl[sample_idx] = torch.tensor(sentence_len, dtype=torch.long)
                        sample_idx += 1
  
                ce_loss.backward()
                self.optim.step()
                batch_loss.append(ce_loss.detach().cpu().item()) 
        # print(self.diff_distribute)
                    
        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))
        self.client_crt_rnd += 1
    
    
    def get_embedding(self,):
        self.model.eval()
        embedding_outputs = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                batch, label = self.adapt_batch(data)
                batch['rt_embedding'] = True
                embedding_outputs.append(torch.mean(self.model(**batch).detach(), dim=0, keepdim=True))
        return embedding_outputs
        

    def run(self):
        self.train()


class Server(BaseServer):
    
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.train_distribute()
        self.uplink()
        self.aggregate_eq()
        if self.args.agg == 'before': self.heterogeneous_agg()
        self.finetune()
        if self.args.agg == 'after': self.heterogeneous_agg()
        self.lr_scheduler()
        self.crt_epoch += 1 
   
   
    def train_distribute(self):
        # == statistic loss for G ==
        if self.is_latent is False:
            self.train_mean = {eq_depth: torch.tensor([0.0, 0.0, 0.0]).to(self.device) for eq_depth in self.eq_depths}
            self.train_std = {eq_depth: torch.tensor([1.0, 1.0, 1.0]).to(self.device) for eq_depth in self.eq_depths}
        else:
            self.clients_embeddings = {eq_depth: [] for eq_depth in self.eq_depths}
            for client in self.sampled_clients:
                self.clients_embeddings[client.eq_depth].extend(client.get_embedding())
            self.clients_embeddings = {eq_depth: torch.cat(self.clients_embeddings[eq_depth], dim=0) for eq_depth in self.eq_depths}
            
            self.train_mean = {eq_depth: torch.mean(self.clients_embeddings[eq_depth], dim=0) for eq_depth in self.eq_depths}
            self.train_std = {eq_depth: None for eq_depth in self.eq_depths}
            # self.train_mean = self.clients_embeddings.mean([0,2], keepdim=True)
            # self.train_std = self.clients_embeddings.std([0,2], keepdim=True)
            del self.clients_embeddings
            # print(self.train_mean, self.train_mean.shape)
            # print(self.train_std, self.train_std.shape)
        
        self.eq_diff = {}
        self.eq_exits_diff = {} # eq_depth 3: [tensor(10*4), tensor(10*4), tensor(10*4)]
        self.eq_y = {}
        self.eq_sl = {}
        for client in self.sampled_clients:
            self.eq_diff.setdefault(client.eq_depth, []).append(client.diff_distribute)
            self.eq_exits_diff.setdefault(client.eq_depth, []).append(client.sample_exits_diff)
            self.eq_y.setdefault(client.eq_depth, []).append(client.sample_y)
            self.eq_sl.setdefault(client.eq_depth, []).append(client.sample_sl)
            
        self.eq_exits_diff = {eq_depth: torch.cat(self.eq_exits_diff[eq_depth], dim=0) for eq_depth in self.eq_depths}
        self.eq_y = {eq_depth: torch.cat(self.eq_y[eq_depth], dim=0) for eq_depth in self.eq_depths}
        self.eq_sl = {eq_depth: torch.cat(self.eq_sl[eq_depth], dim=0) for eq_depth in self.eq_depths}

        for eq_depth in self.eq_depths:
            diff_distribute = [sum(column) for column in zip(*self.eq_diff[eq_depth])]
            diff_distribute = [diff/sum(diff_distribute) for diff in diff_distribute]
            self.eq_diff[eq_depth] = diff_distribute
   
   
    def lr_scheduler(self,):
        # == decay lr for generator & global model ==
        
        for eq_depth in self.eq_depths:
            optimizer = torch.optim.Adam(params=self.generators[eq_depth][0].parameters(), lr=self.g_lr * (self.s_gamma ** self.round))
            self.generators[eq_depth][1] = optimizer
            
            optimizer = torch.optim.SGD(params=self.models[eq_depth][0].parameters(), lr=self.kd_lr * (self.s_gamma ** self.round), weight_decay=1e-3)
            self.models[eq_depth][1] = optimizer
        
        self.sw_optim = torch.optim.Adam(self.sw_net.parameters(), lr=1e-3 * (self.s_gamma ** self.round))
        
    
    def kd_criterion(self, pred, teacher):
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=1)
        T=3
        _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
        return _kld
    
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None, eq_exits=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model, eq_exits)
        
        self.global_model = self.eq_model[max(self.eq_depths)]
        # self.dm = self.eq_model[min(self.eq_depths)]
        # self.dm_policy = self.eq_policy[max(self.eq_depths)]
        self.dm = copy.deepcopy(self.eq_model[max(self.eq_depths)] )
        self.dm_policy = self.eq_policy[max(self.eq_depths)]
        self.clients_embeddings = []
        # == args ==
        self.g_lr, self.g_y, self.g_div, self.g_diff, self.g_gap, self.g_skip, self.g_begin = args.g_lr, args.g_y, args.g_div, args.g_diff, args.g_gap, args.g_skip, args.g_begin
        self.kd_lr, self.kd_gap, self.kd_response_ratio, self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_skip, self.kd_begin = args.kd_lr, args.kd_gap, args.kd_response_ratio, args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_skip, args.kd_begin
        self.s_epoches, self.g_n_iters, self.kd_n_iters = args.s_epoches, args.g_n_iters, args.kd_n_iters
        self.s_gamma = self.args.s_gamma

        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
        self.diff_criterion = nn.MSELoss()
        
        self.is_latent = args.is_latent
        args.exits = self.global_model.config.exits
        
        # == train for generators (each exit has a generator) & eq_models ==
        self.generators = {}
        self.models = {}
        for eq_depth in self.eq_depths:
            generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
            optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr)
            self.generators[eq_depth] = [generator, optimizer]
            
            optimizer = torch.optim.SGD(params=self.eq_model[eq_depth].parameters(), lr=self.kd_lr, weight_decay=1e-3)
            self.models[eq_depth] = [self.eq_model[eq_depth], optimizer]
        self.p=self.args.exit_p
        
        # global model for clients
        for client in self.clients:
            client.server = self
        
        self.max_exit_num = len(self.eq_exits[max(self.eq_depths)])
        self.sw_net = MLP_tanh(input_size=3, output_size=1, hidden_size=100).to(self.device)
        self.sw_optim = torch.optim.Adam(self.sw_net.parameters(), lr=1e-3)
        
        
    def get_batch(self, gen_latent, y_input):
        batch = {}
        if 'cifar' in self.args.dataset or 'svhn' in self.args.dataset or 'imagenet' in self.args.dataset or 'speechcmds' in self.args.dataset:
            batch['pixel_values'] = gen_latent
        else:
            batch['input_ids'] = gen_latent
            batch['attention_mask'] = y_input[1]
        return batch
     
     
    def get_conditional(self, ):
        for g in self.generators.values():
            g[0].to(self.device)
            g[0].train()
        diff_g = {}
        exits_diff_g = {}
        y_input_g = {}
        for eq_depth in self.eq_depths:
            # == new y based y_distribute ==
            attend_eq = [eq_depth]
            # y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            # y_distribute = [y/sum(y_distribute) for y in y_distribute]
            
            # y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # TODO diff now is uniform distribution from 0 to 9, should be changed to client's own difficulty distribution
            diff_distribute = [sum(column) for column in zip(*[[diff*self.eq_num[eq] for diff in self.eq_diff[eq]] for eq in attend_eq])]
            diff_distribute = [diff/sum(diff_distribute) for diff in diff_distribute]
            # print(f'eq_depth{eq_depth}:{diff_distribute}')
            
            diff = torch.tensor(random.choices(range(len(diff_distribute)), weights=diff_distribute, k=self.args.s_bs), dtype=torch.long).to(self.device)
            diff_g[eq_depth] = diff
            random_idxs = torch.randint(0, self.eq_exits_diff[eq_depth].shape[0], (self.args.s_bs,))
            exits_diff_g[eq_depth] = self.eq_exits_diff[eq_depth][random_idxs]
            y_input = self.eq_y[eq_depth][random_idxs].view(-1)
            
            if self.args.dataset in GLUE:
                # TODO two classes for GLUE
                # y_sl_distribute = {y:[sum(column) for column in zip(*[[sl*self.eq_num[eq] for sl in self.eq_y_sl[eq][y]] for eq in attend_eq])] for y in range(0,2)}
                # y_sl_distribute = {y: [sl/sum(y_sl_distribute[y]) for sl in y_sl_distribute[y]] for y in range(0, 2)}

                sentence_lens = self.eq_sl[eq_depth][random_idxs]
                attention_mask = ()
                for i in range(self.args.s_bs):
                    # y = y_input.cpu().tolist()[i]
                    # sentence_len = torch.tensor(random.choices(range(len(y_sl_distribute[y])), weights=y_sl_distribute[y], k=1), dtype=torch.long)
                    
                    # sentence_len = int(sentence_lens[i].cpu().item())
                    sentence_len = 64
                    mask = torch.zeros(128)
                    mask[:sentence_len] = 1
                    attention_mask += (mask.to(self.device), )
                attention_mask = torch.stack(attention_mask)
                y_input_g[eq_depth] = (y_input, attention_mask)
            else: y_input_g[eq_depth] = (y_input, )
                
        return exits_diff_g, diff_g, y_input_g
    
    
    def d_loss(self, gen_latent, y_input, exits_diff):
        dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
        dm_exits_logits = self.dm_policy(dm_exits_logits)
        batch_size = dm_exits_logits[0].shape[0]
        exits_diff_preds = torch.zeros(batch_size, 1).to(self.device)
        for sample_index in range(batch_size):
            diff_pred, exits_diff_pred = difficulty_measure([dm_exits_logits[0][sample_index]], y_input[0][sample_index], metric=self.args.dm, rt_exits_diff=True)
            exits_diff_preds[sample_index] = exits_diff_pred
        
        diff_loss = self.diff_criterion(exits_diff_preds, exits_diff)
        diff_loss = self.g_diff * diff_loss
        return diff_loss


    def y_loss(self, gen_latent, y_input, model, policy, target_probs, exits_num):
        t_exits_logits, t_exits_feature = model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
        t_exits_logits = policy(t_exits_logits)
        t_selected_index_list = exit_policy(exits_num, t_exits_logits, target_probs)
        ce_loss = 0.0
        batch_size = t_exits_logits[0].shape[0]
        for exit_idx, selected_index in enumerate(t_selected_index_list):
            exit_logits = t_exits_logits[exit_idx][selected_index]
            labels = y_input[0][selected_index].long()
            ce_loss += self.ce_criterion(exit_logits, labels) * len(selected_index)
        ce_loss = self.g_y * ce_loss / batch_size
        
        # for exit_logits in t_exits_logits:
        #     ce_loss += self.ce_criterion(exit_logits, y_input[0].long()) * (1/exits_num)
            
        return ce_loss, t_exits_logits, t_exits_feature, t_selected_index_list
    
    
    def diff_distance(self, s_diff_exits, all_diff, sample_index):
        diff, exits_diff = all_diff        
        exits_dis = torch.zeros(len(s_diff_exits)).to(self.device)
        if self.args.sw == 'learn':
            exits_diff = exits_diff[sample_index]
            t_diff = exits_diff
            for i, s_diff in enumerate(s_diff_exits):
                # exits_dis[i] = self.sw_net(torch.abs(t_diff - torch.mean(s_diff, dim=0)))
                exits_dis[i] = self.sw_net(torch.cat((t_diff.unsqueeze(0), torch.mean(s_diff, dim=0, keepdim=True), torch.var(s_diff, dim=0, keepdim=True)), dim=1))
        else:
            diff = diff[sample_index]
            t_diff = diff
            for i, s_diff in enumerate(s_diff_exits):
                exits_dis[i] = F.pairwise_distance(t_diff, torch.mean(s_diff))
        return exits_dis/sum(exits_dis)
    
    
    def gap_loss(self, diff, y_input, selected_index_list, eq_depth, t, s, direction='sl'):
        t_exits_logits, t_exits_feature = t
        s_exits_logits, s_exits_feature = s
        # diff: 1, exits_diff: 4
        diff, exits_diff = diff
        gap_loss = 0.0
        
        # exit policy     
        if direction == 'sl':
            t_exits_num = len(self.eq_exits[eq_depth])
            s_exits_num = len(self.eq_exits[self.eq_depths[self.eq_depths.index(eq_depth)+1]])
        else:
            t_exits_num = len(self.eq_exits[eq_depth])
            s_exits_num = len(self.eq_exits[self.eq_depths[self.eq_depths.index(eq_depth)-1]])
        
        target_probs = calc_target_probs(s_exits_num)[self.p-1]
        s_selected_index_list = exit_policy(s_exits_num, s_exits_logits, target_probs)
        # get diff distribution of each global model exit: diff_exits
        s_diff_exits = []
        for i in range(s_exits_num):
            s_selected_index = s_selected_index_list[i]
            if self.args.sw == 'learn': s_diff_exits.append(exits_diff[s_selected_index])
            else: s_diff_exits.append(diff.float()[s_selected_index])
        
        sum = 0
        # == diff based weight == 
        # for sample 19, samples_distance[19] = [0.2,0.4,0.1,0.3] distance to global exits difficulty distribution
        if direction == 'sl':
            for t_exit_idx in range(t_exits_num):
                if self.args.kd_join == 'last':
                    if t_exit_idx != t_exits_num-1: continue
                t_selected_index = selected_index_list[t_exit_idx]
                weight_t_exits = torch.zeros(s_exits_num).to(self.device)
                for sample_index in t_selected_index:
                    sample_distance = self.diff_distance(s_diff_exits, (diff, exits_diff), sample_index)
                    for s_exit_idx in range(s_exits_num):
                        weight_t_exits[s_exit_idx] = weight_t_exits[s_exit_idx] + sample_distance[s_exit_idx]
                weight_t_exits = F.softmax(-weight_t_exits, dim=0)
                max_weight = weight_t_exits.max()
                weight_t_exits = (weight_t_exits == max_weight).float() if self.args.sw_type == 'hard' else weight_t_exits
                # print(f'eq{eq_depth}_exit{t_exit_idx}:', ["{:.2f}".format(x) for x in weight_t_exits.cpu()])
                t_logits, t_feature = t_exits_logits[t_exit_idx][t_selected_index], t_exits_feature[t_exit_idx][t_selected_index]
                for s_exit_idx in range(s_exits_num):
                    s_logits, s_feature = s_exits_logits[s_exit_idx][t_selected_index], s_exits_feature[s_exit_idx][t_selected_index]
                    t_y = y_input[t_selected_index]
                    sum += s_logits.shape[0]
                    
                    # == ce loss ==
                    gap_ce_loss = 0.0
                    if 'ce' in self.args.loss_type:
                        s, t = s_logits, t_logits.detach()
                        gap_ce_loss = weight_t_exits[s_exit_idx] * self.ce_criterion(s, t_y) * s.shape[0]
                    
                    # == kd gap loss == 
                    gap_kd_loss = 0.0
                    if 'kd' in self.args.loss_type:
                        if direction == 'sl':
                            s, t = s_feature, t_feature.detach()
                            dist_loss = self.kd_dist_ratio*self.dist_criterion(s, t)
                            angle_loss = self.kd_angle_ratio*self.angle_criterion(s, t)
                            dark_loss = self.kd_dark_ratio*self.dark_criterion(s, t)
                            gap_kd_loss = weight_t_exits[s_exit_idx]*(dist_loss + angle_loss + dark_loss) * s.shape[0]
                            
                            # s, t = s_logits, t_logits.detach()
                            # gap_kd_loss = weight_t_exits[s_exit_idx]* self.kd_response_ratio*self.kd_criterion(s, t) * s.shape[0] 
                        else:   
                            s, t = s_logits, t_logits.detach()
                            gap_kd_loss = weight_t_exits[s_exit_idx]* self.kd_response_ratio*self.kd_criterion(s, t) * s.shape[0] 
                            
                    gap_loss += gap_ce_loss + gap_kd_loss
        else:
            for s_exit_idx in range(s_exits_num):
                if s_exit_idx != s_exits_num-1: continue
                e_t_logits = 0
                for t_exit_idx in range(t_exits_num):
                    if t_exit_idx < s_exit_idx: continue
                    t_selected_index = selected_index_list[s_exit_idx]
                    t_logits, t_feature = t_exits_logits[t_exit_idx][t_selected_index], t_exits_feature[t_exit_idx][t_selected_index]
                    e_t_logits += t_logits
                e_t_logits = e_t_logits / (t_exits_num-s_exits_num+1)
                
                s_logits, s_feature = s_exits_logits[s_exit_idx][t_selected_index], s_exits_feature[s_exit_idx][t_selected_index]
                t_y = y_input[t_selected_index]
                sum += s_logits.shape[0]
                
                s, t = s_logits, e_t_logits.detach()
                gap_kd_loss = self.kd_response_ratio*self.kd_criterion(s, t) * s_logits.shape[0]
                gap_loss += gap_kd_loss
             
        gap_loss = gap_loss / sum
        return gap_loss
    
    
    def finetune(self):
        self.s_epoches = int(sum(self.eq_batch_num.values())/len(self.eq_batch_num.values())/self.kd_n_iters) if self.args.adaptive_epoches else self.s_epoches
        # == train generator & global model ==
        for _ in range(self.s_epoches):
            # == train Difficulty-Conditional Generators ==
            exits_diff_g, diff_g, y_input_g = self.get_conditional()

            if self.crt_epoch % self.g_skip == 0 and self.crt_epoch >= self.g_begin:
                self.train_generators(exits_diff_g, y_input_g)
                
            # == train global model utilize generators ==
            if self.crt_epoch % self.kd_skip == 0 and self.crt_epoch >= self.kd_begin:
                self.progressive_train_model(diff_g, exits_diff_g, y_input_g)
        self.dm.load_state_dict(self.eq_model[self.max_depth].state_dict())
    
    
    def train_generators(self, exits_diff_g, y_input_g):
        for g in self.generators.values():
            g[0].to(self.device)
            g[0].train()
        for model in self.models.values():
            model[0].to(self.device)
            model[0].eval() 
        self.sw_net.to(self.device)
        self.sw_net.eval()
        
        # == train generators ==
        for eq_depth, g in self.generators.items():
            self.update_generator(g, eq_depth, self.g_n_iters, exits_diff_g, y_input_g)
    
    
    def update_generator(self, g, eq_depth, n_iters, exits_diff_g, y_input_g):
        # if eq_depth != 12: return
        DIFF_LOSS, CE_LOSS, GAP_LOSS, DIV_LOSS, STT_LOSS = 0, 0, 0, 0, 0
        
        generator = g[0]
        optimizer = g[1]
        t_exits_num = len(self.eq_exits[eq_depth])
        target_probs = calc_target_probs(t_exits_num)[self.p-1]
        # print(target_probs)
        
        for _ in range(n_iters):
            optimizer.zero_grad()
            y_input, exits_diff = y_input_g[eq_depth], exits_diff_g[eq_depth]
            eps = torch.rand((y_input[0].shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)
            gen_latent = g[0](y_input, eps, exits_diff) if self.args.diff_generator else g[0](y_input, eps)
            
            # == LOSS for div sst for G ==
            div_loss = self.g_div * generator.diversity_loss(eps, gen_latent)
            # stt_loss = self.g_gap * generator.statistic_loss(gen_latent, self.train_mean[eq_depth], self.train_std[eq_depth])
            
            # == Loss for diff utilize global model ==
            diff_loss = self.d_loss(gen_latent, y_input, exits_diff) if self.args.diff_generator else torch.tensor(0).to(self.device)
            
            # == Loss for y_input utilize eq_depth super-local model ==             
            ce_loss, t_exits_logits, t_exits_feature, t_selected_index_list = self.y_loss(gen_latent, y_input, self.eq_model[eq_depth], self.eq_policy[eq_depth], target_probs, t_exits_num)
            
            # == Loss for gap == TODO gap LOSS SL & SLS
            gap_loss = 0.0
            # if 'sl' in self.args.kd_direction:
            #     if eq_depth != max(self.eq_depths):                
            #         diff = (exits_diff[:,0], exits_diff)
            #         eq_idx = self.eq_depths.index(eq_depth)
            #         s_model = self.models[self.eq_depths[eq_idx+1]][0]
            #         s_policy = self.eq_policy[self.eq_depths[eq_idx+1]]
            #         s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            #         s_exits_logits = s_policy(s_exits_logits)
            #         gap_loss += self.g_gap * self.gap_loss(diff, y_input[0], t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature))    
            #     else: gap_loss += torch.tensor(0).to(self.device)
            # if 'ls' in self.args.kd_direction:
            #     if eq_depth != min(self.eq_depths):
            #         diff = (exits_diff[:,0], exits_diff)
            #         eq_idx = list(reversed(self.eq_depths)).index(eq_depth)
            #         s_model = self.models[list(reversed(self.eq_depths))[eq_idx+1]][0]
            #         s_policy = self.eq_policy[list(reversed(self.eq_depths))[eq_idx+1]]
            #         s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            #         s_exits_logits = s_policy(s_exits_logits)
            #         gap_loss += self.g_gap * self.gap_loss(diff, y_input[0], t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature), direction='ls')    
            #     else: gap_loss += torch.tensor(0).to(self.device)
                    
            
            # == total loss for backward ==
            loss = ce_loss + diff_loss - gap_loss + div_loss
            loss.backward() # avoid generated data lost in graph
                
            DIFF_LOSS += diff_loss
            CE_LOSS += ce_loss
            GAP_LOSS += gap_loss
            DIV_LOSS += div_loss
            # STT_LOSS += stt_loss
            
            optimizer.step()
        # print(f'============{eq_depth} Super-local Model============')
        # print(f'ce_loss:{CE_LOSS/n_iters:.2f}, div_loss: {DIV_LOSS/n_iters:.2f}, diff_loss: {DIFF_LOSS/n_iters:.2f}, gap_loss: {GAP_LOSS/n_iters:.2f}')
    

    def progressive_train_model(self, diff_g, exits_diff_g, y_input_g):
        # == finetune eq model , multi teacher to teach each exit ==
        for g in self.generators.values():
            g[0].eval()
        for model in self.models.values():
            model[0].to(self.device)
            model[0].train()
        self.sw_net.to(self.device)
        self.sw_net.train()
        
        self.progressive_update_model(self.kd_n_iters, diff_g, exits_diff_g, y_input_g)

    
    def progressive_update_model(self, n_iters, diff_g, exits_diff_g, y_input_g):
        
        # == finetune eq model
        Losses = []
        for _ in range(n_iters):
            for model in self.models.values():
                model[1].zero_grad()
            self.sw_optim.zero_grad()
        
            # == super-sub model teach eq model ==
            Loss = 0.0
            
            sl_Loss = 0.0
            if 'sl' in self.args.kd_direction:
                teacher_sum = sum([self.eq_num[eq_depth] for eq_depth in self.eq_depths[:-1]])
                for idx, eq_depth in enumerate(self.eq_depths):
                    if eq_depth == max(self.eq_depths): continue
                    t_model = self.eq_model[self.eq_depths[idx]]
                    s_model = self.eq_model[self.eq_depths[idx+1]]
                    t_policy = self.eq_policy[self.eq_depths[idx]]
                    s_policy = self.eq_policy[self.eq_depths[idx+1]]
                    
                    with torch.no_grad():
                        y_input, diff, exits_diff = y_input_g[eq_depth], diff_g[eq_depth], exits_diff_g[eq_depth]
                        eps = torch.rand((y_input[0].shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)
                        gen_latent = self.generators[eq_depth][0](y_input, eps, exits_diff).detach()

                        if self.args.diff_generator:
                            diff = (diff, exits_diff)
                        else:
                            batch_size = y_input[0].shape[0]
                            diff_preds = torch.zeros(batch_size, 1).to(self.device)
                            exits_diff_preds = torch.zeros(batch_size, self.max_exit_num).to(self.device)
                            dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                            dm_exits_logits = self.dm_policy(dm_exits_logits)
                            for sample_index in range(batch_size):
                                diff_pred, exits_diff = difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], y_input[0][sample_index], metric=self.args.dm, rt_exits_diff=True)
                                diff_preds[sample_index] = diff_pred
                                exits_diff_preds[sample_index] = exits_diff
                            diff = (diff_preds, exits_diff_preds)
                    
                        t_exits_num = len(self.eq_exits[eq_depth])
                        target_probs = calc_target_probs(t_exits_num)[self.p-1]
                        t_exits_logits, t_exits_feature = t_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                        t_exits_logits = t_policy(t_exits_logits)
                        t_selected_index_list = exit_policy(exits_num=t_exits_num, exits_logits=t_exits_logits, target_probs=target_probs)
                    
                    s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                    s_exits_logits = s_policy(s_exits_logits)
                    sl_Loss += self.kd_gap * self.eq_num[eq_depth]/teacher_sum * self.gap_loss(diff, y_input[0], t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature))
                
            ls_Loss = 0.0
            if 'ls' in self.args.kd_direction:
                for idx, eq_depth in enumerate(reversed(self.eq_depths)):
                    if eq_depth == min(self.eq_depths): continue
                    t_model = self.eq_model[list(reversed(self.eq_depths))[idx]]
                    s_model = self.eq_model[list(reversed(self.eq_depths))[idx+1]]
                    t_policy = self.eq_policy[list(reversed(self.eq_depths))[idx]]
                    s_policy = self.eq_policy[list(reversed(self.eq_depths))[idx+1]]
                    s_depth = list(reversed(self.eq_depths))[idx+1]
                    
                    with torch.no_grad():
                        y_input, diff, exits_diff = y_input_g[s_depth], diff_g[s_depth], exits_diff_g[s_depth]
                        eps = torch.rand((y_input[0].shape[0], self.generators[s_depth][0].noise_dim)).to(self.device)
                        gen_latent = self.generators[s_depth][0](y_input, eps, exits_diff).detach()

                        if self.args.diff_generator:
                            diff = (diff, exits_diff)
                        else:
                            batch_size = y_input[0].shape[0]
                            diff_preds = torch.zeros(batch_size, 1).to(self.device)
                            exits_diff_preds = torch.zeros(batch_size, self.max_exit_num).to(self.device)
                            dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                            dm_exits_logits = self.dm_policy(dm_exits_logits)
                            for sample_index in range(batch_size):
                                diff_pred, exits_diff = difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], y_input[0][sample_index], metric=self.args.dm, rt_exits_diff=True)
                                diff_preds[sample_index] = diff_pred
                                exits_diff_preds[sample_index] = exits_diff
                            diff = (diff_preds, exits_diff_preds)
                    
                        t_exits_num = len(self.eq_exits[eq_depth])
                        target_probs = calc_target_probs(t_exits_num)[self.p-1]
                        t_exits_logits, t_exits_feature = t_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                        t_exits_logits = t_policy(t_exits_logits)
                        t_selected_index_list = exit_policy(exits_num=t_exits_num, exits_logits=t_exits_logits, target_probs=target_probs)
                    
                    s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                    s_exits_logits = s_policy(s_exits_logits)
                    s_exits_num = len(self.eq_exits[list(reversed(self.eq_depths))[idx+1]])
                    s_selected_index_list = exit_policy(exits_num=s_exits_num, exits_logits=s_exits_logits, target_probs=target_probs)
                    ls_Loss += self.kd_gap * self.gap_loss(diff, y_input[0], s_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature), direction='ls')
                  
            Loss = sl_Loss + ls_Loss  
            Loss.backward()
            
            for model in self.models.values():
                model[1].step()
            self.sw_optim.step()
            Losses.append(Loss.cpu().item())
            
        # print(f'Losses: {Losses}')
        

    def sample(self):
        super().sample()
        self.larger_eq_total_num = {eq_depth: sum([self.eq_num[i] for i in self.eq_depths if i >= eq_depth]) for eq_depth in self.eq_depths}
        self.eq_batch_num = {eq_depth: sum([client.batch_num for client in self.sampled_eq_clients[eq_depth]])/(len(self.sampled_eq_clients[eq_depth])) for eq_depth in self.eq_depths}


    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self.eq_model[client.eq_depth])


    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params_eq = {}
        for idx, eq_depth in enumerate(self.eq_depths):
            self.received_params_eq[eq_depth] = [client.weight * client.model.parameters_to_tensor() for client in self.sampled_eq_clients[eq_depth]]

        self.received_params = ()
        for idx, submodel_depth in enumerate(self.eq_depths):
            self.received_params += ([client.model.parameters_to_tensor(is_split=True)[idx] * client.submodel_weights[submodel_depth]
                                for client in self.sampled_submodel_clients[submodel_depth]],)
        
        self.uplink_policy()
        
        
    def aggregate_eq(self):
        assert (len(self.sampled_clients) > 0)
        
        self.aggregate_policy()
        
        for eq_depth in self.eq_depths:
            self.eq_model[eq_depth].tensor_to_parameters(sum(self.received_params_eq[eq_depth]))
            
        # avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        # avg_tensor = torch.cat(avg_eq_tensor, 0)
        # self.global_model.tensor_to_parameters(avg_tensor)
        
        # self.dm.tensor_to_parameters(self.global_model.parameters_to_tensor(is_split=True)[0])
        
        
    def heterogeneous_agg(self):
        depth_weighted_tensor:Dict[int: torch.tensor] = {}
        eq_tensors = {}
        for eq_depth in self.eq_depths:
            eq_model = self.eq_model[eq_depth]
            tensors = eq_model.parameters_to_tensor(is_split=True)
            eq_tensors[eq_depth] = tensors
            for idx in range(len(tensors)):
                depth_weighted_tensor[idx] = depth_weighted_tensor.get(idx, 0.0) + tensors[idx] * self.eq_num[eq_depth]/self.larger_eq_total_num[self.eq_depths[idx]]
        
        aggregated_tensors = list(depth_weighted_tensor.values())
        for idx, eq_depth in enumerate(self.eq_depths):
            aggregated_tensor = torch.cat(aggregated_tensors[:idx+1], 0)
            eq_model: BaseModule = self.eq_model[eq_depth]
            eq_model.tensor_to_parameters(aggregated_tensor)
            
        
    def save_model(self, model_save_path, generator_save_path):
        self.global_model.save_model(model_save_path)
        
        generator_save_path = '.'.join(generator_save_path.split('.')[:-1])
        for i, g in self.generators.items():
            g_model = g[0]
            generator_save_path_i = f'{generator_save_path}_{i}.pth'
            g_model.save_model(generator_save_path_i)
        