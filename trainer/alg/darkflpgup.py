import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import os
import copy

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy, difficulty_measure
from utils.modelload.model import BaseModule
from torch.utils.data import ConcatDataset
from trainer.policy.l2w import MLP_tanh

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def add_args(parser):
    parser.add_argument('--is_latent', default=False, type=bool)
    parser.add_argument('--is_feature', default='False', type=str)
    
    parser.add_argument('--s_epoches', default=10, type=int)
    
    parser.add_argument('--kd_skip', default=1, type=int)
    parser.add_argument('--kd_begin', default=0, type=int)
    parser.add_argument('--kd_lr', default=5e-2, type=float)
    parser.add_argument('--kd_response_ratio', default=3, type=float)
    parser.add_argument('--kd_dist_ratio', default=5, type=float)
    parser.add_argument('--kd_angle_ratio', default=10, type=float)
    parser.add_argument('--kd_dark_ratio', default=0, type=float)
    parser.add_argument('--kd_n_iters', default=5, type=int)
    parser.add_argument('--gap_kd_lambda', default=1, type=float)
    
    parser.add_argument('--g_skip', default=1, type=int)
    parser.add_argument('--g_begin', default=0, type=int)
    parser.add_argument('--g_lr', default=1e-2, type=float)
    parser.add_argument('--g_y', default=1, type=float)
    parser.add_argument('--g_div', default=1, type=float)
    parser.add_argument('--g_gap', default=1, type=float)
    parser.add_argument('--g_diff', default=1, type=float)
    parser.add_argument('--g_n_iters', default=1, type=int)

    parser.add_argument('--kd_direction', default='sl', type=str)
    parser.add_argument('--kd_join', default='last', type=str, help='last: only last exit of teacher can teach student model\'s exits')
    parser.add_argument('--kd_knowledge', default='relation', type=str)
    parser.add_argument('--agg', default='after', type=str)
    
    parser.add_argument('--loss_type', default='ce-kd', type=str)
    parser.add_argument('--dm', default='loss', type=str)
    parser.add_argument('--diff_client_gap', default=2, type=int)
    
    parser.add_argument('--sw', default='learn', type=str, help='how to get weight for students [learn | distance]')
    parser.add_argument('--sw_type', default='soft', type=str, help='weight [soft | hard]')
    return parser




class Client(BaseClient):
    
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        super().__init__(id, args, dataset, model, depth, exits)
        self.diff_distribute = None
        self.client_crt_rnd = 0
    
    def train(self):
        
        # eval diff distribution
        if self.client_crt_rnd % self.args.diff_client_gap == 0:    
            self.diff_distribute = [0 for _ in range(5)]
            for idx, data in enumerate(self.loader_train):
                batch, label = self.adapt_batch(data)
                with torch.no_grad():
                    dm_exits_logits, dm_exits_feature = self.server.dm(**batch, rt_feature=True)
                    dm_exits_logits = self.server.dm_policy(dm_exits_logits)
                    for sample_index in range(label.shape[0]):
                        diff = int(difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], label[sample_index], metric=self.args.dm).cpu().item())
                        self.diff_distribute[diff] += 1
        else:
            self.diff_distribute = self.diff_distribute
                        
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()

                batch, label = self.adapt_batch(data)
                
                if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                    self.policy.train_meta(self.model, batch, label, self.optim)

                exits_ce_loss, _ = self.policy.train(self.model, batch, label)
                ce_loss = sum(exits_ce_loss)
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
        self.get_rawdata()
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
        for client in self.sampled_clients:
            self.eq_diff.setdefault(client.eq_depth, []).append(client.diff_distribute)
        
        for eq_depth in self.eq_depths:
            diff_distribute = [sum(column) for column in zip(*self.eq_diff[eq_depth])]
            diff_distribute = [diff/sum(diff_distribute) for diff in diff_distribute]
            self.eq_diff[eq_depth] = diff_distribute
   
   
    def lr_scheduler(self,):
        # == decay lr for generator & global model ==
        
        for eq_depth in self.eq_depths:
            optimizer = torch.optim.Adam(params=self.generators[eq_depth][0].parameters(), lr=self.g_lr * (self.gamma ** self.round))
            self.generators[eq_depth][1] = optimizer
            
            optimizer = torch.optim.SGD(params=self.models[eq_depth][0].parameters(), lr=self.kd_lr * (self.gamma ** self.round), weight_decay=1e-3)
            self.models[eq_depth][1] = optimizer
        
        self.sw_optim = torch.optim.Adam(self.sw_net.parameters(), lr=1e-3 * (self.gamma ** self.round))
        
    
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
        self.dm = self.eq_model[max(self.eq_depths)]
        self.dm_policy = self.eq_policy[max(self.eq_depths)]
        self.clients_embeddings = []
        # == args ==
        self.is_feature = args.is_feature
        self.g_lr, self.g_y, self.g_div, self.g_diff, self.g_gap, self.g_skip, self.g_begin = args.g_lr, args.g_y, args.g_div, args.g_diff, args.g_gap, args.g_skip, args.g_begin
        self.kd_lr, self.kd_response_ratio, self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_skip, self.kd_begin = args.kd_lr, args.kd_response_ratio, args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_skip, args.kd_begin
        self.s_epoches, self.g_n_iters, self.kd_n_iters = args.s_epoches, args.g_n_iters, args.kd_n_iters
        self.gamma = 0.99       

        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
        self.diff_criterion = nn.MSELoss()
        
        self.is_latent = args.is_latent
        
        # == train for generators (each exit has a generator) & eq_models ==
        self.generators = {}
        self.models = {}
        for eq_depth in self.eq_depths:
            generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
            optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr)
            self.generators[eq_depth] = [generator, optimizer]
            
            optimizer = torch.optim.SGD(params=self.eq_model[eq_depth].parameters(), lr=self.kd_lr, weight_decay=1e-3)
            self.models[eq_depth] = [self.eq_model[eq_depth], optimizer]
        self.p=30
        
        # global model for clients
        for client in self.clients:
            client.server = self
        
        self.max_exit_num = len(self.eq_exits[max(self.eq_depths)])
        self.sw_net = MLP_tanh(input_size=self.max_exit_num, output_size=1, hidden_size=100).to(self.device)
        self.sw_optim = torch.optim.Adam(self.sw_net.parameters(), lr=1e-3)
        
        
    def get_rawdata(self):
        self.eq_loader = {}
        self.eq_dataset = {}
        for eq_depth, clients in self.sampled_eq_clients.items():
            eq_dataset = []
            for client in clients:
                eq_dataset.append(client.dataset_train)    
            self.eq_dataset[eq_depth] = ConcatDataset(eq_dataset)
            self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
     
     
    def get_batch(self, gen_latent, y_input):
        batch = {}
        if 'cifar' in self.args.dataset:
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
        y_input_g = {}
        for eq_depth in self.eq_depths:
            # == new y based y_distribute ==
            attend_eq = [eq_depth]
            y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # TODO diff now is uniform distribution from 0 to 9, should be changed to client's own difficulty distribution
            diff_distribute = [sum(column) for column in zip(*[[diff*self.eq_num[eq] for diff in self.eq_diff[eq]] for eq in attend_eq])]
            diff_distribute = [diff/sum(diff_distribute) for diff in diff_distribute]
            # print(f'eq_depth{eq_depth}:{diff_distribute}')
            
            diff = torch.tensor(random.choices(range(len(diff_distribute)), weights=diff_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            diff_g[eq_depth] = diff
            
            if self.args.dataset in GLUE:
                # TODO two classes for GLUE
                y_sl_distribute = {y:[sum(column) for column in zip(*[[sl*self.eq_num[eq] for sl in self.eq_y_sl[eq][y]] for eq in attend_eq])] for y in range(0,2)}
                y_sl_distribute = {y: [sl/sum(y_sl_distribute[y]) for sl in y_sl_distribute[y]] for y in range(0, 2)}
            
                attention_mask = ()
                for i in range(self.args.bs):
                    y = y_input.cpu().tolist()[i]
                    sentence_len = torch.tensor(random.choices(range(len(y_sl_distribute[y])), weights=y_sl_distribute[y], k=1), dtype=torch.long)
                    mask = torch.zeros(128)
                    mask[:sentence_len] = 1
                    attention_mask += (mask.to(self.device), )
                attention_mask = torch.stack(attention_mask)
                y_input_g[eq_depth] = (y_input, attention_mask)
            else: y_input_g[eq_depth] = (y_input, )
                
        return diff_g, y_input_g
    
    
    def d_loss(self, gen_latent, y_input, diff):
        dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
        dm_exits_logits = self.eq_policy[min(self.eq_depths)](dm_exits_logits)
        batch_size = dm_exits_logits[0].shape[0]
        diff_preds = torch.zeros(batch_size, 1).to(self.device)
        for sample_index in range(batch_size):
            diff_pred = difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], y_input[0][sample_index], metric=self.args.dm)
            diff_preds[sample_index] = diff_pred
        
        diff_loss = self.diff_criterion(diff_preds, diff.float().view(batch_size, -1))
        diff_loss = self.g_diff * diff_loss
        return diff_loss


    def y_loss(self, gen_latent, y_input, model, policy, target_probs, exits_num):
        t_exits_logits, t_exits_feature = model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
        t_exits_logits = policy(t_exits_logits)
        t_selected_index_list = exit_policy(exits_num, t_exits_logits, target_probs)
        ce_loss = 0.0
        # batch_size = t_exits_logits[0].shape[0]
        # for exit_idx, selected_index in enumerate(t_selected_index_list):
        #     exit_logits = t_exits_logits[exit_idx][selected_index]
        #     labels = y_input[0][selected_index].long()
        #     ce_loss += self.ce_criterion(exit_logits, labels) * len(selected_index)
        # ce_loss = self.g_y * ce_loss / batch_size
        
        for exit_logits in t_exits_logits:
            ce_loss += self.ce_criterion(exit_logits, y_input[0].long()) * (1/exits_num)
            
        return ce_loss, t_exits_logits, t_exits_feature, t_selected_index_list
    
    
    def diff_distance(self, s_diff_exits, all_diff, sample_index):
        diff, exits_diff = all_diff
        diff = diff[sample_index]
        exits_diff = exits_diff[sample_index]
        
        exits_dis = torch.zeros(len(s_diff_exits)).to(self.device)
        if self.args.sw == 'learn':
            t_diff = exits_diff
            for i, s_diff in enumerate(s_diff_exits):
                exits_dis[i] = self.sw_net(torch.abs(t_diff - torch.mean(s_diff, dim=0)))
        else:
            t_diff = diff
            for i, s_diff in enumerate(s_diff_exits):
                exits_dis[i] = F.pairwise_distance(t_diff, torch.mean(s_diff))
        return exits_dis/sum(exits_dis)
    
    
    def gap_loss(self, diff, y_input, t_selected_index_list, eq_depth, t, s, direction='sl'):
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
            selected_index = s_selected_index_list[i]
            if self.args.sw == 'learn': s_diff_exits.append(exits_diff[selected_index])
            else: s_diff_exits.append(diff.float()[selected_index])
        
        sum = 0
        for t_exit_idx in range(t_exits_num):
            if self.args.kd_join == 'last':
                if t_exit_idx != t_exits_num-1: continue
            selected_index = t_selected_index_list[t_exit_idx]
            
            # == diff based weight == 
            # for sample 19, samples_distance[19] = [0.2,0.4,0.1,0.3] distance to global exits difficulty distribution
            weight_t_exits = torch.zeros(s_exits_num).to(self.device)
            for sample_index in selected_index:
                sample_distance = self.diff_distance(s_diff_exits, (diff, exits_diff), sample_index)
                for s_exit_idx in range(s_exits_num):
                    weight_t_exits[s_exit_idx] = weight_t_exits[s_exit_idx] + sample_distance[s_exit_idx]
            weight_t_exits = F.softmax(-weight_t_exits, dim=0)
            max_weight = weight_t_exits.max()
            weight_t_exits = (weight_t_exits == max_weight).float() if self.args.sw_type == 'hard' else weight_t_exits
            # print(f'eq{eq_depth}_exit{t_exit_idx}:', ["{:.2f}".format(x) for x in weight_t_exits.cpu()])

            t_selected_index = t_selected_index_list[t_exit_idx]
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
                    else:   
                        s, t = s_logits, t_logits.detach()
                        s, t = F.normalize(s, p=2, dim=1), F.normalize(t, p=2, dim=1)
                        gap_kd_loss = weight_t_exits[s_exit_idx]* F.mse_loss(s, t) * s.shape[0] 
                gap_loss += gap_ce_loss + self.args.gap_kd_lambda*gap_kd_loss
                
        gap_loss = self.g_gap * gap_loss / sum
        return gap_loss
    
    
    def finetune(self):
        # == train generator & global model ==
        for _ in range(self.s_epoches):
            # == train Difficulty-Conditional Generators ==
            diff_g, y_input_g = self.get_conditional()

            # if self.crt_epoch % self.g_skip == 0 and self.crt_epoch >= self.g_begin:
            #     self.train_generators(diff_g, y_input_g)
                
            # == train global model utilize generators ==
            if self.crt_epoch % self.kd_skip == 0 and self.crt_epoch >= self.kd_begin:
                self.progressive_train_model(diff_g, y_input_g)
                    
    
    def train_generators(self, diff_g, y_input_g):
        for g in self.generators.values():
            g[0].to(self.device)
            g[0].train()
        for model in self.models.values():
            model[0].to(self.device)
            model[0].eval() 
        
        # == train generators ==
        for eq_depth, g in self.generators.items():
            print(f'============{eq_depth} Super-local Model============')
            self.update_generator(g, eq_depth, self.g_n_iters, diff_g, y_input_g)
    
    
    def update_generator(self, g, eq_depth, n_iters, diff_g, y_input_g):
        # if eq_depth != 12: return
        DIFF_LOSS, CE_LOSS, GAP_LOSS, DIV_LOSS, STT_LOSS = 0, 0, 0, 0, 0
        
        generator = g[0]
        optimizer = g[1]
        t_exits_num = len(self.eq_exits[eq_depth])
        target_probs = calc_target_probs(t_exits_num)[self.p-1]
        # print(target_probs)
        
        for _ in range(n_iters):
            optimizer.zero_grad()
            diff, y_input = diff_g[eq_depth], y_input_g[eq_depth]
            eps = torch.rand((y_input[0].shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)
            gen_latent = g[0](diff, y_input, eps)
            
            # == LOSS for div sst for G ==
            div_loss = self.g_div * generator.diversity_loss(eps, gen_latent)
            # stt_loss = self.g_gap * generator.statistic_loss(gen_latent, self.train_mean[eq_depth], self.train_std[eq_depth])
            
            # == Loss for diff utilize global model ==
            diff_loss = self.d_loss(gen_latent, y_input, diff)
            
            # == Loss for y_input utilize eq_depth super-local model ==             
            ce_loss, t_exits_logits, t_exits_feature, t_selected_index_list = self.y_loss(gen_latent, y_input, self.eq_model[eq_depth], self.eq_policy[eq_depth], target_probs, t_exits_num)
            
            # == Loss for gap ==
            # s_exits_logits, s_exits_feature = self.global_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            # s_exits_logits = self.eq_policy[max(self.eq_depths)](s_exits_logits)
            # gap_loss = 10 * self.gap_loss(diff, t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature))    
            gap_loss = torch.tensor(0).to(self.device)
            
            # == total loss for backward ==
            loss = ce_loss + diff_loss - gap_loss + div_loss
            loss.backward() # avoid generated data lost in graph
            
            DIFF_LOSS += diff_loss
            CE_LOSS += ce_loss
            GAP_LOSS += gap_loss
            DIV_LOSS += div_loss
            # STT_LOSS += stt_loss
            
            optimizer.step()
        print(f'ce_loss:{CE_LOSS.cpu().item()/n_iters:.2f}, div_loss: {DIV_LOSS.cpu().item()/n_iters:.2f}, diff_loss: {DIFF_LOSS.cpu().item()/n_iters:.2f}, gap_loss: {GAP_LOSS.cpu().item()/n_iters:.2f}')
    

    def progressive_train_model(self, diff_g, y_input_g):
        # == finetune eq model , multi teacher to teach each exit ==
        for g in self.generators.values():
            g[0].eval()
        for model in self.models.values():
            model[0].to(self.device)
            model[0].train()
        self.sw_net.to(self.device)
        self.sw_net.train()
        
        self.progressive_update_model(self.kd_n_iters, diff_g, y_input_g)

    
    def progressive_update_model(self, n_iters, diff_g, y_input_g):
        
        # == finetune eq model
        Losses = []
        for _ in range(n_iters):
            for model in self.models.values():
                model[1].zero_grad()
            self.sw_optim.zero_grad()
        
            # == super-sub model teach eq model ==
            Loss = 0.0
            
            sl_Loss = 0.0
            if self.args.kd_direction == 'sl' or self.args.kd_direction == 'sls':
                for idx, eq_depth in enumerate(self.eq_depths):
                    if eq_depth == max(self.eq_depths): continue
                    t_model = self.eq_model[self.eq_depths[idx]]
                    s_model = self.eq_model[self.eq_depths[idx+1]]
                    t_policy = self.eq_policy[self.eq_depths[idx]]
                    s_policy = self.eq_policy[self.eq_depths[idx+1]]
                    
                    # y_input, diff = y_input_g[eq_depth], diff_g[eq_depth]
                    # eps = torch.rand((y_input[0].shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)
                    # gen_latent = self.generators[eq_depth][0](diff, y_input, eps).detach()
                    
                    while True:
                        with torch.no_grad():
                            try:
                                data = next(self.eq_loader[eq_depth])
                            except StopIteration :
                                self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
                                data = next(self.eq_loader[eq_depth])
                            # self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
                            
                            batch, label = self.adapt_batch(data)
                            gen_latent = batch['pixel_values']
                            y_input = label                
                            dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=False, rt_feature=True)
                            dm_exits_logits = self.dm_policy(dm_exits_logits)
                            batch_size = y_input.shape[0]
                            diff_preds = torch.zeros(batch_size, 1).to(self.device)
                            exits_diff_preds = torch.zeros(batch_size, self.max_exit_num).to(self.device)
                            for sample_index in range(batch_size):
                                diff_pred, exits_diff = difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], y_input[sample_index], metric=self.args.dm, rt_exits_diff=True)
                                diff_preds[sample_index] = diff_pred
                                exits_diff_preds[sample_index] = exits_diff
                            diff = (diff_preds, exits_diff_preds)
                            if batch_size == self.args.bs: break
                
                    t_exits_num = len(self.eq_exits[eq_depth])
                    target_probs = calc_target_probs(t_exits_num)[self.p-1]
                    with torch.no_grad():
                        t_exits_logits, t_exits_feature = t_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                        t_exits_logits = t_policy(t_exits_logits)
                        t_selected_index_list = exit_policy(exits_num=t_exits_num, exits_logits=t_exits_logits, target_probs=target_probs)
                    
                    s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True, frozen=False)
                    s_exits_logits = s_policy(s_exits_logits)
                    sl_Loss += self.gap_loss(diff, y_input, t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature))
                
            ls_Loss = 0.0
            # if self.args.kd_direction == 'ls' or self.args.kd_direction == 'sls':
            #     for idx, eq_depth in enumerate(reversed(self.eq_depths)):
            #         if eq_depth == min(self.eq_depths): continue
            #         t_model = self.eq_model[list(reversed(self.eq_depths))[idx]]
            #         s_model = self.eq_model[list(reversed(self.eq_depths))[idx+1]]
                    
            #         # y_input, diff = y_input_g[eq_depth], diff_g[eq_depth]
            #         # eps = torch.rand((y_input[0].shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)
            #         # gen_latent = self.generators[eq_depth][0](diff, y_input, eps).detach()
                    
            #         while True:
            #             with torch.no_grad():
            #                 try:
            #                     data = next(self.eq_loader[eq_depth])
            #                 except StopIteration :
            #                     self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
            #                     data = next(self.eq_loader[eq_depth])
            #                 # self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
                            
            #                 batch, label = self.adapt_batch(data)
            #                 gen_latent = batch['pixel_values']
            #                 y_input = label                
            #                 dm_exits_logits, dm_exits_feature = self.dm(**self.get_batch(gen_latent, y_input), is_latent=False, rt_feature=True)
            #                 batch_size = y_input.shape[0]
            #                 diff_preds = torch.zeros(batch_size, 1).to(self.device)
            #                 for sample_index in range(batch_size):
            #                     diff_preds[sample_index] = difficulty_measure([dm_exits_logits[i][sample_index] for i in range(len(dm_exits_logits))], y_input[sample_index], metric=self.args.dm)
            #                 diff = diff_preds
            #                 if batch_size == self.args.bs: break
                
            #         t_exits_num = len(self.eq_exits[eq_depth])
            #         target_probs = calc_target_probs(t_exits_num)[self.p-1]
            #         with torch.no_grad():
            #             t_exits_logits, t_exits_feature = t_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            #             t_selected_index_list = exit_policy(exits_num=t_exits_num, exits_logits=t_exits_logits, target_probs=target_probs)
                    
            #         s_exits_logits, s_exits_feature = s_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True, frozen=False)
            #         ls_Loss += self.gap_loss(diff, t_selected_index_list, eq_depth, (t_exits_logits, t_exits_feature), (s_exits_logits, s_exits_feature), direction='ls')
                    
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
        