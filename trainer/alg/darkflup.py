import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.utils.data import ConcatDataset

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy


def add_args(parser):
    parser.add_argument('--is_latent', default=False, type=bool)
    parser.add_argument('--is_feature', default=False, type=bool)
    
    parser.add_argument('--s_epoches', default=10, type=int)
    
    parser.add_argument('--kd_gap', default=1, type=int)
    parser.add_argument('--kd_begin', default=0, type=int)
    parser.add_argument('--kd_lr', default=1e-3, type=float)
    parser.add_argument('--kd_response_ratio', default=3, type=float)
    parser.add_argument('--kd_dist_ratio', default=5, type=float)
    parser.add_argument('--kd_angle_ratio', default=10, type=float)
    parser.add_argument('--kd_dark_ratio', default=0, type=float)
    parser.add_argument('--kd_n_iters', default=1, type=int)
    
    
    parser.add_argument('--g_gap', default=1, type=int)
    parser.add_argument('--g_begin', default=0, type=int)
    parser.add_argument('--g_alpha', default=1, type=float)
    parser.add_argument('--g_beta', default=1, type=float)
    parser.add_argument('--g_eta', default=1, type=float)
    parser.add_argument('--g_gamma', default=10, type=float)
    parser.add_argument('--g_lr', default=1e-2, type=float)
    parser.add_argument('--g_n_iters', default=1, type=int)
    return parser




class Client(BaseClient):
    
    def train(self):
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
        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))
    
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
        self.aggregate()
        self.finetune()
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
   
   
    def lr_scheduler(self,):
        # == decay lr for generator & global model ==
        for g in self.generators.values():
            g[2].step()
        self.global_scheduler.step()
   
    
    def kd_criterion(self, pred, teacher):
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=1)
        T=3
        _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
        return _kld
    
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None, eq_exits=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model, eq_exits)
        
        self.clients_embeddings = []
        # == args ==
        self.is_feature = args.is_feature
        self.g_lr, self.g_alpha, self.g_beta, self.g_eta, self.g_gamma, self.g_gap, self.g_begin = args.g_lr, args.g_alpha, args.g_beta, args.g_eta, args.g_gamma, args.g_gap, args.g_begin
        self.kd_lr, self.kd_response_ratio, self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_gap, self.kd_begin = args.kd_lr, args.kd_response_ratio, args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_gap, args.kd_begin
        self.s_epoches, self.g_n_iters, self.kd_n_iters = args.s_epoches, args.g_n_iters, args.kd_n_iters
        self.gamma = 0.99
        
        # == train for global model ==
        self.global_optimizer = torch.optim.Adam(params=self.global_model.parameters(), lr=self.kd_lr)
        self.global_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.global_optimizer, gamma=self.gamma)
        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        
        self.is_latent = args.is_latent
        # == train for generators (each exit has a generator) ==
        self.generators = {}
        for eq_depth in self.eq_depths:
            generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
            optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
            self.generators[eq_depth] = [generator, optimizer, lr_scheduler]
        self.p=30
    
    
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
     
     
    def get_gen_latent(self, ):
        for g in self.generators.values():
            g[0].to(self.device)
            g[0].train()
        diff_g = {}
        y_input_g = {}
        eps_g = {}
        for eq_depth in self.eq_depths:
            # == new y based y_distribute ==
            attend_eq = [eq_depth]
            y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # TODO diff now is uniform distribution from 0 to 9, should be changed to client's own difficulty distribution
            diff_distribute = [1/10 for i in range(10)]
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
                
            eps_g[eq_depth] = torch.rand((y_input.shape[0], self.generators[eq_depth][0].noise_dim)).to(self.device)

        return diff_g, y_input_g, eps_g
    
    
    def finetune(self):
        # == train generator & global model ==
        for _ in range(self.s_epoches):
            # == train Difficulty-Conditional Generators ==
            # diff_g, y_input_g, eps_g = self.get_gen_latent()
            # gen_latent_g = {}
            # for eq_depth in self.eq_depths:
            #     gen_latent = self.generators[eq_depth][0](diff_g[eq_depth], y_input_g[eq_depth], eps_g[eq_depth])
            #     gen_latent_g[eq_depth] = gen_latent      
            # if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
            #     self.train_generators(diff_g, y_input_g, gen_latent_g, eps_g)
            
            # for eq_depth, y_input in y_input_g.items():
            #     if isinstance(y_input, tuple): y_input_g[eq_depth] = tuple(tensor.detach() for tensor in y_input)
                
            # # == train global model utilize generators ==
            # gen_latent_g = {}
            # for eq_depth in self.eq_depths:
            #     gen_latent = self.generators[eq_depth][0](diff_g[eq_depth], y_input_g[eq_depth], eps_g[eq_depth])
            #     gen_latent_g[eq_depth] = gen_latent.detach()
            if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                self.finetune_global_model()
                
    
    def train_generators(self, diff_g, y_input_g, gen_latent_g, eps_g):
        for g in self.generators.values():
            g[0].to(self.device)
            g[0].train()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        self.global_model.eval()
        
        # == train generators ==
        for eq_depth, g in self.generators.items():
            print(f'============{eq_depth} Super-local Model============')
            self.update_generator(g, eq_depth, self.g_n_iters, diff_g, y_input_g, gen_latent_g, eps_g)
    
    
    def update_generator(self, g, eq_depth, n_iters, diff_g, y_input_g, gen_latent_g, eps_g):
        
        DIFF_LOSS, CE_LOSS, DIV_LOSS, STT_LOSS = 0, 0, 0, 0
        
        generator = g[0]
        optimizer = g[1]
        exits_num = len(self.eq_exits[eq_depth])
        target_probs = calc_target_probs(exits_num)[self.p-1]
        # print(target_probs)
        
        for i in range(n_iters):
            optimizer.zero_grad()
            diff, y_input, gen_latent, eps = diff_g[eq_depth], y_input_g[eq_depth], gen_latent_g[eq_depth], eps_g[eq_depth]
            
            # == LOSS for div sst for G ==
            div_loss = self.g_beta * generator.diversity_loss(eps, gen_latent)
            stt_loss = self.g_gamma * generator.statistic_loss(gen_latent, self.train_mean[eq_depth], self.train_std[eq_depth])
            # div_loss, stt_loss = 0.0, 0.0
            
            # == Loss for diff utilize global model ==
            exits_logits, exits_feature = self.global_model(**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            exits_logits = self.eq_policy[max(self.eq_depths)](exits_logits)
            batch_size = exits_logits[0].shape[0]
            diff_preds = torch.zeros(batch_size, 1).to(self.device)
            for sample_index in range(batch_size):
                last_logits = exits_logits[-1][sample_index].unsqueeze(0)
                diff_pred = 0
                for exit_idx in range(len(exits_logits)):
                    exit_logits = exits_logits[exit_idx][sample_index].unsqueeze(0)
                    diff_pred += nn.functional.cosine_similarity(exit_logits, last_logits, dim=1)
                diff_preds[sample_index] = (1-diff_pred/len(exits_logits))*10
            
            # print(diff_preds.device, diff.device)
            diff_loss = self.mse_criterion(diff_preds, diff.float().view(batch_size, -1))
            diff_loss = self.g_eta * diff_loss
            
            # == Loss for y_input utilize eq_depth super-local model == 
            exits_logits, exits_feature = self.eq_model[eq_depth](**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
            exits_logits = self.eq_policy[eq_depth](exits_logits)
            selected_index_list = exit_policy(exits_num, exits_logits, target_probs)
            ce_loss = 0.0
            for exit_idx, selected_index in enumerate(selected_index_list):
                exit_logits = exits_logits[exit_idx][selected_index]
                labels = y_input[0][selected_index].long()
                ce_loss += self.ce_criterion(exit_logits, labels) * len(selected_index)
            ce_loss = self.g_alpha * ce_loss / batch_size
            
            # == total loss for backward ==
            loss = ce_loss + div_loss + stt_loss + diff_loss
            loss.backward(retain_graph=True) if i < n_iters - 1 else loss.backward() # avoid generated data lost in graph
            
            DIFF_LOSS += diff_loss
            CE_LOSS += ce_loss
            DIV_LOSS += div_loss
            STT_LOSS += stt_loss
            
            optimizer.step()
        print(f'ce_loss:{CE_LOSS/n_iters}, div_loss: {DIV_LOSS/n_iters}, stt_loss: {STT_LOSS/n_iters}, diff_loss: {DIFF_LOSS/n_iters}')
    
        
    def difficulty_measure(self, exits_logits, label=None, metric='loss'):
        with torch.no_grad():
            exits_logits = self.eq_policy[max(self.eq_depths)](exits_logits)
            if metric == 'loss':
                exits_loss = ()
                loss_func = nn.CrossEntropyLoss()
                for i, logits in enumerate(exits_logits):
                    exits_loss += (loss_func(logits, label),)
                diff_pred = sum(exits_loss)
                
            elif metric == 'confidence':
                confidences = 0
                for logits in exits_logits:
                    confidence, _ = F.softmax(logits, dim=0).max(dim=0, keepdim=False)
                    confidences += confidence
                diff_pred = (1-confidences/len(exits_logits))*10
                
            elif metric == 'cosine':
                last_logits = exits_logits[-1].unsqueeze(0)
                diff_pred = 0
                for logits in exits_logits:
                    exit_logits = logits.unsqueeze(0)
                    diff_pred += nn.functional.cosine_similarity(exit_logits, last_logits, dim=1)
                diff_pred = (1-diff_pred/len(exits_logits))*10
            return diff_pred
        
        
    def finetune_global_model(self):
        # == finetune global model , multi teacher to teach each exit ==
        for g in self.generators.values():
            g[0].eval()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        self.global_model.train()
        
        y_input_g, gen_latent_g = {}, {}
        with torch.no_grad():
            for eq_depth in self.eq_depths:
                try:
                    data = next(self.eq_loader[eq_depth])
                except StopIteration :
                    self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
                    data = next(self.eq_loader[eq_depth])
                # self.eq_loader[eq_depth] = iter(torch.utils.data.DataLoader(self.eq_dataset[eq_depth], batch_size=self.args.bs, shuffle=False, collate_fn=None))
                
                batch, label = self.adapt_batch(data)
                gen_latent_g[eq_depth] = batch['pixel_values']
                y_input_g[eq_depth] = label

        self.teach_global_model(self.generators, self.kd_n_iters, y_input_g, gen_latent_g)

    
    def teach_global_model(self, gs, n_iters, y_input_g, gen_latent_g):
        

        def diff_distance(local_diff, global_diff_exits):
            exits_dis = torch.zeros(len(global_diff_exits)).to(self.device)
            for i, global_diff in enumerate(global_diff_exits):
                exits_dis[i] = F.pairwise_distance(local_diff, torch.mean(global_diff))
            return exits_dis/sum(exits_dis)

        
        # == finetune global model
        Losses = []
        for _ in range(n_iters):
            self.global_optimizer.zero_grad()

            # == diff measure used global model ==
            # == exit policy for generator with all pesudo data ==
            global_n_exits = len(self.eq_exits[max(self.eq_depths)])
            global_diff_exits = [[] for _ in range(global_n_exits)]
            sample_num, diff_g = 0, {}
            for eq_depth in self.eq_depths:
                y_input, gen_latent = y_input_g[eq_depth], gen_latent_g[eq_depth]
                
                exits_num = len(self.eq_exits[eq_depth])
                exits_logits, exits_feature = self.global_model(**self.get_batch(gen_latent, y_input), is_latent=False, rt_feature=True)
                
                batch_size = y_input.shape[0]
                diff_preds = torch.zeros(batch_size, 1).to(self.device)
                for sample_index in range(batch_size):
                    diff_preds[sample_index] = self.difficulty_measure([exits_logits[i][sample_index] for i in range(len(exits_logits))], label=y_input[sample_index], metric='loss')
                diff_g[eq_depth] = diff_preds
                
                target_probs = calc_target_probs(exits_num)[self.p-1]
                selected_index_list = exit_policy(exits_num=exits_num, exits_logits=exits_logits[:exits_num], target_probs=target_probs)
                for exit_idx in range(exits_num):
                    selected_index = selected_index_list[exit_idx]
                    sample_num += len(selected_index)
                    global_diff_exits[exit_idx].append(diff_preds[selected_index])
            global_diff_exits = [torch.cat(list, dim=0) for list in global_diff_exits]
            # print(global_diff_exits)
            avg_diff = 0.0
            for global_diff in global_diff_exits: avg_diff += (sum(global_diff).cpu().item()) / sample_num
            print(avg_diff, [torch.mean(global_diff).cpu().item() for global_diff in global_diff_exits])

            # == super-sub model teach global model ==   
            Loss = 0.0
            for eq_depth in self.eq_depths:
                gen_latent, y_input, diff = gen_latent_g[eq_depth], y_input_g[eq_depth], diff_g[eq_depth]
                exits_num = len(self.eq_exits[eq_depth])
                target_probs = calc_target_probs(exits_num)[self.p-1]
                exits_logits, exits_feature = self.eq_model[eq_depth](**self.get_batch(gen_latent, y_input), is_latent=self.is_latent, rt_feature=True)
                exits_logits = self.eq_policy[eq_depth](exits_logits)
                selected_index_list = exit_policy(exits_num=exits_num, exits_logits=exits_logits, target_probs=target_probs)
                # print(eq_depth, selected_index_list)
                
                for exit_idx in range(exits_num):
                    # weight based difficulty distribution
                    samples_distance = {} # for sample 19, samples_distance[19] = [0.2,0.4,0.1,0.3] distance to global exits difficulty distribution
                    selected_index = selected_index_list[exit_idx]
                    weight_t_exits = torch.zeros(global_n_exits).to(self.device)
                    for sample_index in selected_index:
                        samples_distance[sample_index] = diff_distance(diff[sample_index].unsqueeze(0), global_diff_exits)
                        for t_exit in range(global_n_exits):
                            weight_t_exits[t_exit] += samples_distance[sample_index][t_exit]
                    weight_t_exits = F.softmax(-weight_t_exits, dim=0)
                    # weight_t_exits = (weight_t_exits == weight_t_exits.max())
                    
                    # print(weight_t_exits)
                    
                    # # hard weight
                    # weight_t_exits = torch.zeros(global_n_exits).to(self.device)
                    # if eq_depth != max(self.eq_depths):
                    #     if exit_idx == len(self.eq_exits[eq_depth])-1:
                    #         weight_t_exits[exit_idx+1] = 1
                            
                    # print(f'eq{eq_depth}_exit{exit_idx}:', ["{:.4f}".format(x) for x in weight_t_exits.cpu()]) if  _==n_iters-1 else None
                        
                    t_y_input = y_input[selected_index]
                    t_gen_latent = gen_latent[selected_index]
                    t_logits = exits_logits[exit_idx][selected_index]
                    t_feature = exits_feature[exit_idx][selected_index]
                    
                    s_exits_logits, s_exits_feature = self.global_model(**self.get_batch(t_gen_latent, t_y_input), is_latent=self.is_latent, rt_feature=True, frozen=True)
                    
                    for s_exit_idx in range(global_n_exits):
                        s_logits = self.eq_policy[max(self.eq_depths)].sf(s_exits_logits[:s_exit_idx+1])
                        s_feature = s_exits_feature[s_exit_idx]
                        if self.is_feature: s, t = s_feature, t_feature
                        else: s, t = s_logits, t_logits
                        Loss += weight_t_exits[s_exit_idx]*(self.kd_dist_ratio*self.dist_criterion(s, t) + self.kd_angle_ratio*self.angle_criterion(s, t) + self.kd_dark_ratio*self.dark_criterion(s, t))
                        # Loss += weight_t_exits[s_exit_idx]*(self.ce_criterion(s, t_y_input))
                        # Loss += weight_t_exits[s_exit_idx]*(self.kd_criterion(s, t))
            
            Loss.backward()

            self.global_optimizer.step()
            Losses.append(Loss.cpu().item())
        print(f'Losses: {Losses}')
        

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
        
        
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        
        self.aggregate_policy()
        
        for eq_depth in self.eq_depths:
            self.eq_model[eq_depth].tensor_to_parameters(sum(self.received_params_eq[eq_depth]))
            
        avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        avg_tensor = torch.cat(avg_eq_tensor, 0)
        self.global_model.tensor_to_parameters(avg_tensor)
        
        
    def save_model(self, model_save_path, generator_save_path):
        self.global_model.save_model(model_save_path)
        
        generator_save_path = '.'.join(generator_save_path.split('.')[:-1])
        for i, g in self.generators.items():
            g_model = g[0]
            generator_save_path_i = f'{generator_save_path}_{i}.pth'
            g_model.save_model(generator_save_path_i)
        