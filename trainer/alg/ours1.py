import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient
from trainer.generator.generator import Generator, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

def add_args(parser):
    parser.add_argument('--kd_gap', default=5, type=int)
    parser.add_argument('--kd_begin', default=10, type=int)
    parser.add_argument('--kd_lr', default=3e-4, type=float)
    parser.add_argument('--kd_dist_ratio', default=1, type=float)
    parser.add_argument('--kd_angle_ratio', default=2, type=float)
    parser.add_argument('--kd_dark_ratio', default=0, type=float)
    parser.add_argument('--kd_n_iters', default=5, type=int)
    parser.add_argument('--kd_epochs', default=10, type=int)
    
    parser.add_argument('--g_gap', default=5, type=int)
    parser.add_argument('--g_begin', default=10, type=int)
    parser.add_argument('--g_alpha', default=1, type=float)
    parser.add_argument('--g_beta', default=1, type=float)
    parser.add_argument('--g_eta', default=1, type=float)
    parser.add_argument('--g_lr', default=3e-4, type=float)
    parser.add_argument('--g_n_iters', default=10, type=int)
    parser.add_argument('--g_epochs', default=10, type=int)
    return parser.parse_args()

class Client(BaseClient):
    def run(self):
        self.train()
    
class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate_train()
        # if self.crt_epoch % 20 == 0 and self.crt_epoch > 0:
        #     self.sample()
        #     self.downlink()
        #     self.client_update()
        #     self.uplink()
        #     self.aggregate_train()
        # else:
        #     BaseServer.sample(self)
        #     BaseServer.downlink(self)
        #     BaseServer.client_update(self)
        #     BaseServer.uplink(self)
        #     BaseServer.aggregate(self)

        self.crt_epoch += 1 
    
    
    def kd_criterion(self, pred, teacher):
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=1)
        T=3
        _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
        return _kld
    
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None, eqs_exits=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model, eqs_exits=eqs_exits)
        
        self.global_model = self.eq_model[max(self.eq_depths)]
        
        # == init eq_models' optimizer, lr_scheduler
        self.eq_model_train = {}
        for eq_depth, eq_model in self.eq_model.items():
            param_optimizer = list(eq_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.kd_lr, correct_bias=False)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
            self.eq_model_train[eq_depth] = (eq_model, optimizer, scheduler)
        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
              
        # == args ==
        self.g_lr, self.g_alpha, self.g_beta, self.g_eta, self.g_gap, self.g_begin = args.g_lr, args.g_alpha, args.g_beta, args.g_eta, args.g_gap, args.g_begin
        self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_gap, self.kd_begin = args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_gap, args.kd_begin
        
        self.g_epochs, self.g_n_iters = args.g_epochs, args.g_n_iters
        self.kd_epochs, self.kd_n_iters = args.kd_epochs, args.kd_n_iters
        
        # == generators ==
        # sl_generators[eq_depth][exit_indx] generate data for eq_depth's last exit to teach larger eq's exit_index-th exit
        self.sl_generators = {}
        for i, eq_depth in enumerate(self.eq_depths):
            if eq_depth != max(self.eq_depths):
                generators = {}
                larger_eq_depth = self.eq_depths[i+1]
                for j in range(len(self.eqs_exits[eq_depth])-1, len(self.eqs_exits[larger_eq_depth])):
                    # generator = Generator(args)
                    generator = Generator_CIFAR(args)
                    optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr,  betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
                    generators[j]  = (generator, optimizer, lr_scheduler)
                self.sl_generators[eq_depth] = generators
        # ls generators    
        self.ls_generators = {}
        for i, eq_depth in enumerate(reversed(self.eq_depths)):
            if eq_depth != min(self.eq_depths):
                generators = {}
                smaller_eq_depth = self.eq_depths[i-1]
                for j in range(len(self.eqs_exits[smaller_eq_depth])-1, len(self.eqs_exits[eq_depth])):
                    generator = Generator_CIFAR(args)
                    optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr,  betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
                    generators[j] = (generator, optimizer, lr_scheduler)
                self.ls_generators[eq_depth] = generators
    
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = {}
        for idx, eq_depth in enumerate(self.eq_depths):
            self.received_params[eq_depth] = [client.weight * client.model.parameters_to_tensor() for client in self.sampled_eq_clients[eq_depth]]
    
    
    def aggregate_train(self,):
        
        # == First: aggregate each eq lonely ==
        for eq_depth in self.eq_depths:
            received_tensor = sum(self.received_params[eq_depth])
            self.eq_model[eq_depth].tensor_to_parameters(received_tensor)
        
        # == Second: small to large ==
        # 1. aggregate params 2. train generator 2. small teach large
        for i, eq_depth in enumerate(self.eq_depths):
            if eq_depth == max(self.eq_depths): break
            
            # == aggregate for eq ==
            self.aggregate_aligned_layers(eq_depth)
            
            # == each KD arrow for specific eq ==
            # == small eq's last exit teach larger eq's deeper exits == 
            for s_exit, g in self.sl_generators[eq_depth].items():
                t = self.eq_model_train[self.eq_depths[i]]
                s = self.eq_model_train[self.eq_depths[i+1]]
                
                # == train generator for each arrow ==
                if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
                    print("=================")
                    print(eq_depth, s_exit)
                    for _ in range(self.g_epochs):
                        self.update_generator(self.g_n_iters, g, eq_depth, t[0], s[0], s_exit, direction='sl')
                    g[2].step()

                # == small eq's last exit teach larger eq's deeper exit == 
                if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                    for _ in range(self.kd_epochs):
                        self.teach_next_model(self.kd_n_iters, g, eq_depth, t, s, s_exit, direction='sl')
                    s[2].step()

        # == Third: large to small ==
        for i, eq_depth in enumerate(reversed(self.eq_depths)):
            if eq_depth == min(self.eq_depths): break

            # == each KD arrow for specific eq ==
            # == larger eq's deeper exits teach smaller eq's last exits ==
            for t_exit, g in self.ls_generators[eq_depth].items():
                t = self.eq_model_train[reversed(self.eq_depths)[i]]
                s = self.eq_model_train[reversed(self.eq_depths)[i+1]]
                 
                 # == train generator for each arrow ==
                if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
                     print("================")
                     print(eq_depth, t_exit)
                     for _ in range(self.g_epochs):
                         self.update_generator(self.g_n_iters, g, eq_depth, t[0], s[0], t_exit, direction='ls')
                
                # == large eq's deeper exits teach smaller eq's last exit ==
                if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                    for _ in range(self.kd_epochs):
                        self.teach_next_model(self.kd_n_iters, g, eq_depth, t, s, t_exit, direction='ls')
            

    def aggregate_aligned_layers(self, eq_depth):
        exits = self.eqs_exits[eq_depth]
        if eq_depth == min(self.eq_depths):
            begin_layer = 0
            end_layer = self.eqs_exits[eq_depth][-2] if len(self.eqs_exits[eq_depth]) > 1 else 0
        elif eq_depth == max(self.eq_depths):
            begin_layer = self.eqs_exits[self.eq_depths[self.eq_depths.index(eq_depth)-1]][-1]+1
            end_layer = max(self.eq_depths)
        else:
            begin_layer = self.eqs_exits[self.eq_depths[self.eq_depths.index(eq_depth)-1]][-1]+1
            end_layer = self.eqs_exits[eq_depth][-2] if len(self.eqs_exits[eq_depth]) > 1 else 0
    
        aligned_layers = [begin_layer, end_layer]
        
        tensors = []
        attend_eqs = self.eq_depths[self.eq_depths.index(eq_depth):]
        total_num = sum([self.eq_num[eq_depth] for eq_depth in attend_eqs])
        for depth in attend_eqs:
            tensors.append(self.eq_model[depth].parameters_to_tensor(layers=aligned_layers)*self.eq_num[depth]/total_num)
        tensor = sum(tensors)
        for depth in attend_eqs:
            self.eq_model[depth].tensor_to_parameters(tensor, layers=aligned_layers)
    
    
    def teach_next_model(self, n_iters, g, t_eq, t, s, g_exit, direction='sl'):
        DIST_LOSS, ANGLE_LOSS, DARK_LOSS, KD_LOSS = 0, 0, 0, 0
        
        generator, t_model, s_model, s_optimizer, s_scheduler = g[0], t[0], s[0], s[1], s[2]

        t_model.eval()
        generator.eval()
        s_model.train()
        
        t_model.to(self.device)
        generator.to(self.device)
        s_model.to(self.device)
        
        for _ in range(n_iters):
            s_optimizer.zero_grad()
            
            y_distribute = self.eq_y[t_eq]
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # == data ==
            gen_latent, eps = generator(y_input, )
            
            if direction == 'sl':
            
                begin_exit = len(t_model.config.exits)-2 if len(t_model.config.exits) > 1 else None
                
                # t_logits = t_model(latent=gen_latent, exit_idxs=(begin_exit, len(t_model.config.exits)-1))
                # s_logits = s_model(latent=gen_latent, exit_idxs=(begin_exit, s_exit))
                
                t_logits = t_model(gen_latent)[-1]
                s_logits = s_model(gen_latent)[g_exit]
                
                dist_loss = self.kd_dist_ratio*self.dist_criterion(s_logits, t_logits)
                angle_loss = self.kd_angle_ratio*self.angle_criterion(s_logits, t_logits)
                dark_loss = self.kd_dark_ratio*self.dark_criterion(s_logits, t_logits)
                
                DIST_LOSS += dist_loss
                ANGLE_LOSS += angle_loss
                DARK_LOSS += dark_loss
                
                loss = dist_loss + angle_loss + dark_loss
            
            elif direction == 'ls':
                
                t_logits = t_model(gen_latent)[g_exit]
                s_logits = s_model(gen_latent)[-1]
                
                kd_loss = self.kd_criterion(s_logits, t_logits)
                
                KD_LOSS += kd_loss
                
                loss = kd_loss
            
            loss.backward()
            s_optimizer.step()
        # s_scheduler.step()
        if direction == 'sl':
            print(f'dist_loss:{DIST_LOSS/n_iters}, angle_loss: {ANGLE_LOSS/n_iters}, dark_loss: {DARK_LOSS/n_iters}')
        else:
            print(f'kd_loss:{KD_LOSS/n_iters}')
        
    
    def update_generator(self, n_iters, g, t_eq, t_model, s_model, g_exit, direction='sl'):        
        
        CE_LOSS, DIV_LOSS, KD_LOSS = 0, 0, 0
        
        generator: Generator= g[0]
        optimizer:torch.optim.optimizer = g[1]
        lr_scheduler = g[2]

        generator.train()
        t_model.eval()
        s_model.eval()
        generator.to(self.device)
        t_model.to(self.device)
        s_model.to(self.device)
        for i in range(n_iters):
            optimizer.zero_grad()
            # == new y based y_distribute ==
            y_distribute = self.eq_y[t_eq]
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # == data ==
            gen_latent, eps = generator(y_input, )
            
            # == div kd ce loss for G ==
            # == div loss for G ==
            div_loss = self.g_beta*generator.diversity_loss(eps, gen_latent)
            
            if direction == 'sl':
            
                begin_exit = len(t_model.config.exits)-2 if len(t_model.config.exits) > 1 else None
                
                # besides s_exit, all_logits len is (s_exits - t_exits + 1) - 1
                all_other_logits = ()
                for end_exit in self.sl_generators[t_eq].keys():
                    if end_exit == g_exit: continue
                    # logits = s_model(latent=gen_latent, exit_idxs=(begin_exit, end_exit))
                    logits = s_model(gen_latent)[end_exit]
                    all_other_logits += (logits, )
                    
                # == ensemble_logits for student exits [batch * hidden_size]
                ensemble_logits = torch.mean(torch.stack(all_other_logits), dim=0)
                
                # == teacher's logits ==
                # t_logits = t_model(latent=gen_latent, exit_idxs=(begin_exit, len(t_model.config.exits)-1))
                t_logits = t_model(gen_latent)[-1]
                
                # == kd_loss for G ==
                kd_loss = self.g_eta*self.kd_criterion(ensemble_logits, t_logits)
                
                # == ce_loss for G ==
                ce_loss = self.g_alpha*self.ce_criterion(t_logits, y_input.view(-1))
            
            elif direction == 'ls':
                
                t_logits = t_model[gen_latent][g_exit]
                s_logits = s_model[gen_latent][-1]
                
                kd_loss = self.g_eta*self.kd_criterion(s_logits, t_logits)
                ce_loss = self.g_alpha*self.ce_criterion(t_logits, y_input.view(-1))
            
            loss = ce_loss + div_loss - kd_loss
            loss.backward()
            
            CE_LOSS += ce_loss
            DIV_LOSS += div_loss
            KD_LOSS += kd_loss
            
            optimizer.step()
        # lr_scheduler.step()
        print(f'ce_loss:{CE_LOSS/n_iters}, div_loss: {DIV_LOSS/n_iters}, kd_loss: {KD_LOSS/n_iters}')
    
            
    def sample(self):
        self.sampled_eq_clients = {}
        
        sample_num = int(self.sample_rate * self.client_num)
        
        check_all_depths_sampled = {}
        while sum(check_all_depths_sampled.values()) != len(self.eq_depths):
            check_all_depths_sampled.clear()
            self.sampled_clients: List[BaseClient] = random.sample(self.clients, sample_num)
            for client in self.sampled_clients:
                check_all_depths_sampled[client.eq_depth] = 1

        for eq_depth in self.eq_depths:
            for client in self.sampled_clients:
                if client.eq_depth == eq_depth:
                    self.sampled_eq_clients.setdefault(eq_depth, []).append(client)
        
        self.eq_num = {}
        for eq_depth, clients in self.sampled_eq_clients.items():
            total_samples = sum(len(client.dataset_train) for client in clients)
            self.eq_num[eq_depth] = total_samples
            for client in clients:
                client.weight = len(client.dataset_train) / total_samples
   
                
    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self.eq_model[client.eq_depth])
    