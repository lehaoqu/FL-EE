import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient
from trainer.generator.generator import Generator, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

def add_args(parser):
    parser.add_argument('--is_latent', default=False, type=bool)
    
    parser.add_argument('--kd_gap', default=1, type=int)
    parser.add_argument('--kd_begin', default=0, type=int)
    parser.add_argument('--kd_lr', default=1e-4, type=float)
    parser.add_argument('--kd_response_ratio', default=3, type=float)
    parser.add_argument('--kd_dist_ratio', default=1, type=float)
    parser.add_argument('--kd_angle_ratio', default=2, type=float)
    parser.add_argument('--kd_dark_ratio', default=0, type=float)
    parser.add_argument('--kd_n_iters', default=10, type=int)
    parser.add_argument('--kd_epochs', default=1, type=int)
    
    parser.add_argument('--g_gap', default=1, type=int)
    parser.add_argument('--g_begin', default=0, type=int)
    parser.add_argument('--g_alpha', default=1, type=float)
    parser.add_argument('--g_beta', default=1, type=float)
    parser.add_argument('--g_eta', default=1, type=float)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--g_n_iters', default=10, type=int)
    parser.add_argument('--g_epochs', default=1, type=int)
    return parser

class Client(BaseClient):
    def run(self):
        self.train()


class Server(BaseServer):
    
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
            self.train_generator()
        if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
            self.finetune()
        self.crt_epoch += 1 
   
    
    def kd_criterion(self, pred, teacher):
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=1)
        T=3
        _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
        return _kld
    
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None, eq_exits=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model, eq_exits)
        
        # == args ==
        self.g_lr, self.g_alpha, self.g_beta, self.g_eta, self.g_gap, self.g_begin = args.g_lr, args.g_alpha, args.g_beta, args.g_eta, args.g_gap, args.g_begin
        self.kd_lr, self.kd_response_ratio, self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_gap, self.kd_begin = args.kd_lr, args.kd_response_ratio, args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_gap, args.kd_begin
        
        self.g_epochs, self.g_n_iters = args.g_epochs, args.g_n_iters
        self.kd_epochs, self.kd_n_iters = args.kd_epochs, args.kd_n_iters
        
        # == train for global model ==
        # param_optimizer = list(self.global_model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #     'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        # ]
        # optimizer = torch.optim.Adam(params=self.global_model.parameters(), lr=self.kd_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
        self.global_optimizer = torch.optim.Adam(params=self.global_model.parameters(), lr=self.kd_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
        self.global_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.global_optimizer, gamma=args.gamma)
        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
        
        self.is_latent = args.is_latent
        # == train for generators (each exit has a generator) ==
        self.generators = []
        for i in range(len(self.eq_exits[max(self.eq_depths)])):
            generator = Generator_CIFAR(args) if self.is_latent is False else Generator(args)
            optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
            self.generators.append((generator, optimizer, lr_scheduler))
            
    
    def train_generator(self):
        for g in self.generators:
            g[0].to(self.device)
            g[0].train()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        
        # == train generators ==
        for exit_idx, g in enumerate(self.generators):
            print(f'============{exit_idx}============')
            for i in range(self.g_epochs):
                self.update_generator(g, exit_idx, self.g_n_iters)
            g[2].step()
    
    
    def update_generator(self, g, exit_idx, n_iters):
        
        CE_LOSS, DIV_LOSS, KD_LOSS = 0, 0, 0
        
        generator = g[0]
        optimizer = g[1]
        
        attend_eq = [eq_depth for eq_depth in self.eq_depths if exit_idx < len(self.eq_exits[eq_depth])]
        
        for i in range(n_iters):
            optimizer.zero_grad()
            
            # == new y based y_distribute ==
            y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # == data ==
            gen_latent, eps = generator(y_input, )
            
            # == div kd ce loss for G ==
            # == div loss for G ==
            div_loss = self.g_beta*generator.diversity_loss(eps, gen_latent)

            # == ensemble logits for attend eq's
            attend_logits = ()
            for eq_depth in attend_eq:
                attend_logits += (self.eq_policy[eq_depth].sf(self.eq_model[eq_depth](gen_latent, stop_exit=exit_idx, is_latent=self.is_latent)) * self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in attend_eq]), )
            attend_logits = sum(attend_logits)
            
            ce_loss = self.g_alpha*self.ce_criterion(attend_logits, y_input.view(-1))
            kd_loss = self.g_eta*torch.zeros(1).to(self.device)
            
            if exit_idx != 0:
                former_attend_eq = [eq_depth for eq_depth in self.eq_depths if exit_idx-1 < len(self.eq_exits[eq_depth])]
                former_attend_logits = ()
                for eq_depth in former_attend_eq:
                    former_attend_logits += (self.eq_policy[eq_depth].sf(self.eq_model[eq_depth](gen_latent, stop_exit=exit_idx-1, is_latent=self.is_latent)) * self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in former_attend_eq]), )
                former_attend_logits = sum(former_attend_logits)
                
                kd_loss = self.g_eta*self.kd_criterion(attend_logits, former_attend_logits)
            
            loss = ce_loss + div_loss - kd_loss
            loss.backward()
            
            CE_LOSS += ce_loss
            DIV_LOSS += div_loss
            KD_LOSS += kd_loss
            
            optimizer.step()
        print(f'ce_loss:{CE_LOSS/n_iters}, div_loss: {DIV_LOSS/n_iters}, kd_loss: {KD_LOSS/n_iters}')
    
        
    def finetune(self,):
        # == finetune global model , multi teacher to teach each exit ==
        for g in self.generators:
            g[0].eval()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        self.global_model.train()
        
        self.teach_global_model(self.generators, self.kd_n_iters)
        self.global_scheduler.step()
    
    
    def teach_global_model(self, gs, n_iters):
        Loss = 0.0
        for _ in range(n_iters):
            self.global_optimizer.zero_grad()
            for s_exit in range(len(self.eq_exits[max(self.eq_depths)])):
                loss = torch.zeros(1).to(self.device)
                t_exits = (s_exit-1, s_exit, s_exit+1)
                
                for t_exit in t_exits:
                    if t_exit >= 0 and t_exit < len(self.eq_exits[max(self.eq_depths)]):
                        # == new y based y_distribute ==
                        attend_eq = [eq_depth for eq_depth in self.eq_depths if t_exit < len(self.eq_exits[eq_depth])]
                        y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
                        y_distribute = [y/sum(y_distribute) for y in y_distribute]
                        y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
                        
                        # == data ==
                        gen_latent, eps = gs[t_exit][0](y_input, )

                        s_logits = self.eq_policy[max(self.eq_depths)].sf(self.global_model(gen_latent, stop_exit=s_exit, is_latent=self.is_latent))
                        
                        attend_logits = ()
                        for eq_depth in attend_eq:
                            attend_logits += (self.eq_policy[eq_depth].sf(self.eq_model[eq_depth](gen_latent, stop_exit=t_exit, is_latent=self.is_latent)) * self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in attend_eq]), )
                        attend_logits = sum(attend_logits)
                        
                        if t_exit >= s_exit:
                            loss += self.kd_response_ratio*self.kd_criterion(s_logits, attend_logits)
                        else:
                            loss += self.kd_dist_ratio*self.dist_criterion(s_logits, attend_logits) + self.kd_angle_ratio*self.angle_criterion(s_logits, attend_logits) + self.kd_dark_ratio*self.dark_criterion(s_logits, attend_logits)
                
                loss.backward()
                Loss += loss.detach().cpu().item()
            self.global_optimizer.step()
        
        print(f'Loss: {Loss/n_iters}')


    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params_eq = {}
        for idx, eq_depth in enumerate(self.eq_depths):
            self.received_params_eq[eq_depth] = [client.weight * client.model.parameters_to_tensor() for client in self.sampled_eq_clients[eq_depth]]

        self.received_params = ()
        for idx, submodel_depth in enumerate(self.eq_depths):
            self.received_params += ([client.model.parameters_to_tensor(is_split=True)[idx] * client.submodel_weights[submodel_depth]
                                for client in self.sampled_submodel_clients[submodel_depth]],)
        
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        
        for eq_depth in self.eq_depths:
            self.eq_model[eq_depth].tensor_to_parameters(sum(self.received_params_eq[eq_depth]))
            
        avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        avg_tensor = torch.cat(avg_eq_tensor, 0)
        self.global_model.tensor_to_parameters(avg_tensor)
        
    def save_model(self, model_save_path, generator_save_path):
        self.global_model.save_model(model_save_path)
        
        generator_save_path = '.'.join(generator_save_path.split('.')[:-1])
        for i, g in enumerate(self.generators):
            g_model = g[0]
            generator_save_path_i = f'{generator_save_path}_{i}.pth'
            g_model.save_model(generator_save_path_i)
        