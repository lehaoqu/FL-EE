import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

def add_args(parser):
    parser.add_argument('--is_latent', default=False, type=bool)
    
    parser.add_argument('--s_epoches', default=5, type=int)
    
    parser.add_argument('--sl', default=1, type=int)
    parser.add_argument('--ls', default=1, type=int)
    
    parser.add_argument('--kd_gap', default=1, type=int)
    parser.add_argument('--kd_begin', default=0, type=int)
    parser.add_argument('--kd_lr', default=5e-3, type=float)
    parser.add_argument('--kd_dist_ratio', default=1, type=float)
    parser.add_argument('--kd_angle_ratio', default=3, type=float)
    parser.add_argument('--kd_dark_ratio', default=0, type=float)
    parser.add_argument('--kd_n_iters', default=5, type=int)
    
    parser.add_argument('--g_gap', default=1, type=int)
    parser.add_argument('--g_begin', default=0, type=int)
    parser.add_argument('--g_alpha', default=1, type=float)
    parser.add_argument('--g_beta', default=1, type=float)
    parser.add_argument('--g_eta', default=1, type=float)
    parser.add_argument('--g_gamma', default=1, type=float)
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
        self.downlink()
        self.client_update()
        self.train_distribute()
        self.uplink()
        self.aggregate_eq()
        self.finetune()
        self.lr_scheduler()

        self.crt_epoch += 1 
    
    
    def train_distribute(self):
        # == statistic loss for G ==
        if self.is_latent is False:
            self.train_mean = torch.tensor([0.0, 0.0, 0.0]).to(self.device)
            self.train_std = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        else:
            self.clients_embeddings = []
            for client in self.sampled_clients:
                self.clients_embeddings.extend(client.get_embedding())
            self.clients_embeddings = torch.cat(self.clients_embeddings, dim=0)
            self.train_mean = self.clients_embeddings.mean([0,2], keepdim=True)
            self.train_std = self.clients_embeddings.std([0,2], keepdim=True)
            del self.clients_embeddings
            # print(self.train_mean, self.train_mean.shape)
            # print(self.train_std, self.train_std.shape)
    
    
    def lr_scheduler(self,):
        for gs in self.sl_generators.values():
            for g in gs.values():
                g[2].step()
        for gs in self.ls_generators.values():
            for g in gs.values():
                g[2].step()
        for eq_model in self.eq_model_train.values():
            eq_model[2].step()
    
    
    def kd_criterion(self, pred, teacher):
        kld_loss = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax = nn.Softmax(dim=1)
        T=3
        _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
        return _kld
    
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None, eq_exits=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model, eq_exits=eq_exits)
        
        # == args ==
        self.g_lr, self.g_alpha, self.g_beta, self.g_eta, self.g_gamma, self.g_gap, self.g_begin = args.g_lr, args.g_alpha, args.g_beta, args.g_eta, args.gamma, args.g_gap, args.g_begin
        self.kd_lr, self.kd_dist_ratio, self.kd_angle_ratio, self.kd_dark_ratio, self.kd_gap, self.kd_begin = args.kd_lr, args.kd_dist_ratio, args.kd_angle_ratio, args.kd_dark_ratio, args.kd_gap, args.kd_begin
        self.s_epoches, self.g_n_iters, self.kd_n_iters = args.s_epoches, args.g_n_iters, args.kd_n_iters
        
        self.global_model = self.eq_model[max(self.eq_depths)]
        self.gamma = 0.99
        
        # == init eq_models' optimizer, lr_scheduler
        self.eq_model_train = {}
        for eq_depth, eq_model in self.eq_model.items():
            # param_optimizer = list(eq_model.named_parameters())
            # no_decay = ['bias', 'gamma', 'beta']
            # optimizer_grouped_parameters = [
            #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            #     'weight_decay_rate': 0.01},
            #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            # ]
            optimizer = torch.optim.SGD(params=eq_model.parameters(), lr=self.kd_lr, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
            self.eq_model_train[eq_depth] = (eq_model, optimizer, scheduler)
        
        # == relation KD loss for small to large ==
        self.dist_criterion = RkdDistance()
        self.angle_criterion = RKdAngle()
        self.dark_criterion = HardDarkRank()
        
        # == ce&kd loss for generator train ==
        self.ce_criterion = nn.CrossEntropyLoss()
        
        self.is_latent = args.is_latent
        # == generators ==
        # sl_generators[eq_depth][exit_indx] generate data for eq_depth's last exit to teach larger eq's exit_index-th exit
        self.sl_generators = {}
        for i, eq_depth in enumerate(self.eq_depths):
            if eq_depth != max(self.eq_depths):
                generators = {}
                larger_eq_depth = self.eq_depths[i+1]
                for j in range(len(self.eq_exits[eq_depth])-1, len(self.eq_exits[larger_eq_depth])):
                    # generator = Generator(args)
                    generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
                    optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr)
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
                    generators[j]  = (generator, optimizer, lr_scheduler)
                self.sl_generators[eq_depth] = generators
        # ls generators    
        self.ls_generators = {}
        for i, eq_depth in enumerate(reversed(self.eq_depths)):
            if eq_depth != min(self.eq_depths):
                generators = {}
                smaller_eq_depth = list(reversed(self.eq_depths))[i+1]
                for j in range(len(self.eq_exits[smaller_eq_depth])-1, len(self.eq_exits[eq_depth])):
                    generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
                    optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
                    generators[j] = (generator, optimizer, lr_scheduler)
                self.ls_generators[eq_depth] = generators
    
    
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = {}
        for idx, eq_depth in enumerate(self.eq_depths):
            self.received_params[eq_depth] = [client.weight * client.model.parameters_to_tensor() for client in self.sampled_eq_clients[eq_depth]]
    
        self.uplink_policy()
    
    
    def aggregate_eq(self):
        # == fed l2w meta_net ==
        self.aggregate_policy()
        
        # == First: aggregate each eq lonely ==
        for eq_depth in self.eq_depths:
            received_tensor = sum(self.received_params[eq_depth])
            self.eq_model[eq_depth].tensor_to_parameters(received_tensor)
    
    
    def get_gen_latent(self, g, t_eq):
        g_model = g[0]
        g_model.to(self.device)
        g_model.train()
        y_distribute = self.eq_y[t_eq]
        y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
        # == data ==
        gen_latent, eps = g[0](y_input, )
        return y_input, gen_latent, eps
        
    
    def finetune(self,):
        for _ in range(self.s_epoches):
    
            # == Second: small to large ==
            # 1. aggregate params 2. train generator 2. small teach large
            for i, eq_depth in enumerate(self.eq_depths):
                if eq_depth == max(self.eq_depths): break
                
                # == aggregate for eq ==
                self.aggregate_aligned_layers(eq_depth)
                
                if self.args.sl == 0: continue
                # == each KD arrow for specific eq ==
                # == small eq's last exit teach larger eq's deeper exits == 
                for s_exit, g in self.sl_generators[eq_depth].items():
                    
                    t_eq = self.eq_depths[i]
                    s_eq = self.eq_depths[i+1]
                    t = self.eq_model_train[t_eq]
                    s = self.eq_model_train[s_eq]
                    y_input, gen_latent, eps = self.get_gen_latent(g, t_eq)
                    
                    
                    # == train generator for each arrow ==
                    if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
                        print("=================")
                        print(eq_depth, s_exit)
                        self.update_generator(self.g_n_iters, g, t_eq, s_eq, t[0], s[0], s_exit, y_input, gen_latent, eps, direction='sl')

                    # == small eq's last exit teach larger eq's deeper exit == 
                    if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                        self.teach_next_model(self.kd_n_iters, g, t_eq, s_eq, t, s, s_exit, y_input.detach(), gen_latent.detach(), eps.detach(), direction='sl')

            # == Third: large to small ==
            for i, eq_depth in enumerate(reversed(self.eq_depths)):
                if eq_depth == min(self.eq_depths): break

                if self.args.ls == 0: continue
                # == each KD arrow for specific eq ==
                # == larger eq's deeper exits teach smaller eq's last exits ==
                for t_exit, g in self.ls_generators[eq_depth].items():
                    t_eq = list(reversed(self.eq_depths))[i]
                    s_eq = list(reversed(self.eq_depths))[i+1]
                    t = self.eq_model_train[t_eq]
                    s = self.eq_model_train[s_eq]
                    y_input, gen_latent, eps = self.get_gen_latent(g, t_eq)
                    
                    # == train generator for each arrow ==
                    if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
                        print("================")
                        print(eq_depth, t_exit)
                        self.update_generator(self.g_n_iters, g, t_eq, s_eq, t[0], s[0], t_exit, y_input, gen_latent, eps, direction='ls')
                    
                    # == large eq's deeper exits teach smaller eq's last exit ==
                    if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                        self.teach_next_model(self.kd_n_iters, g, t_eq, s_eq, t, s, t_exit, y_input.detach(), gen_latent.detach(), eps.detach(), direction='ls')
                

    def aggregate_aligned_layers(self, eq_depth):
        exits = self.eq_exits[eq_depth]
        if eq_depth == min(self.eq_depths):
            begin_layer = 0
            end_layer = self.eq_exits[eq_depth][-2] if len(self.eq_exits[eq_depth]) > 1 else 0
        elif eq_depth == max(self.eq_depths):
            begin_layer = self.eq_exits[self.eq_depths[self.eq_depths.index(eq_depth)-1]][-1]+1
            end_layer = max(self.eq_depths)
        else:
            begin_layer = self.eq_exits[self.eq_depths[self.eq_depths.index(eq_depth)-1]][-1]+1
            end_layer = self.eq_exits[eq_depth][-2] if len(self.eq_exits[eq_depth]) > 1 else 0
    
        aligned_layers = [begin_layer, end_layer]
        
        tensors = []
        attend_eqs = self.eq_depths[self.eq_depths.index(eq_depth):]
        total_num = sum([self.eq_num[eq_depth] for eq_depth in attend_eqs])
        for depth in attend_eqs:
            tensors.append(self.eq_model[depth].parameters_to_tensor(layers=aligned_layers)*self.eq_num[depth]/total_num)
        tensor = sum(tensors)
        for depth in attend_eqs:
            self.eq_model[depth].tensor_to_parameters(tensor, layers=aligned_layers)
    
    
    def update_generator(self, n_iters, g, t_eq, s_eq, t_model, s_model, g_exit, y_input, gen_latent, eps, direction='sl'):        
        
        CE_LOSS, DIV_LOSS, KD_LOSS, STT_LOSS = 0, 0, 0, 0
        
        t_policy = self.eq_policy[t_eq]
        s_policy = self.eq_policy[s_eq]
        
        generator = g[0]
        optimizer:torch.optim.optimizer = g[1]

        generator.train()
        t_model.eval()
        s_model.eval()
        generator.to(self.device)
        t_model.to(self.device)
        s_model.to(self.device)
        for i in range(n_iters):
            optimizer.zero_grad()
            
            # == div kd ce loss for G ==
            # == div loss for G ==
            div_loss = self.g_beta*generator.diversity_loss(eps, gen_latent)
            stt_loss = self.g_gamma*generator.statistic_loss(gen_latent, self.train_mean, self.train_std)
            
            if direction == 'sl':
                begin_exit = len(t_model.config.exits)-2 if len(t_model.config.exits) > 1 else None
                
                # besides s_exit, all_logits len is (s_exits - t_exits + 1) - 1
                all_other_logits = ()
                for end_exit in self.sl_generators[t_eq].keys():
                    if end_exit == g_exit: continue
                    # logits = s_model(latent=gen_latent, exit_idxs=(begin_exit, end_exit))
                    logits = s_policy.sf(s_model(gen_latent, stop_exit=end_exit, is_latent=self.is_latent))
                    all_other_logits += (logits, )
                    
                # == ensemble_logits for student exits [batch * hidden_size]
                ensemble_logits = torch.mean(torch.stack(all_other_logits), dim=0)
                
                # == teacher's logits ==
                # t_logits = t_model(latent=gen_latent, exit_idxs=(begin_exit, len(t_model.config.exits)-1))
                t_logits = t_policy.sf(t_model(gen_latent, is_latent=self.is_latent))
                
                # == kd_loss for G ==
                kd_loss = self.g_eta*self.kd_criterion(ensemble_logits, t_logits)
                
                # == ce_loss for G ==
                ce_loss = self.g_alpha*self.ce_criterion(t_logits, y_input.view(-1))
            
            elif direction == 'ls':
                t_logits = t_policy.sf(t_model(gen_latent, stop_exit=g_exit, is_latent=self.is_latent))
                s_logits = s_policy.sf(s_model(gen_latent, is_latent=self.is_latent))
                
                kd_loss = self.g_eta*self.kd_criterion(s_logits, t_logits)
                ce_loss = self.g_alpha*self.ce_criterion(t_logits, y_input.view(-1))
            
            loss = ce_loss + div_loss - kd_loss + stt_loss
            loss.backward(retain_graph=True) if i < n_iters-1 else loss.backward() 
            
            CE_LOSS += ce_loss
            DIV_LOSS += div_loss
            KD_LOSS += kd_loss
            STT_LOSS += stt_loss
            
            optimizer.step()
        print(f'ce_loss:{CE_LOSS/n_iters}, div_loss: {DIV_LOSS/n_iters}, kd_loss: {KD_LOSS/n_iters}, stt_loss: {STT_LOSS/n_iters}')
       
    
    def teach_next_model(self, n_iters, g, t_eq, s_eq, t, s, g_exit, y_input, gen_latent, eps, direction='sl'):
        DIST_LOSS, ANGLE_LOSS, DARK_LOSS, KD_LOSS = 0, 0, 0, 0

        t_policy = self.eq_policy[t_eq]
        s_policy = self.eq_policy[s_eq]
                
        generator, t_model, s_model, s_optimizer, s_scheduler = g[0], t[0], s[0], s[1], s[2]

        t_model.eval()
        generator.eval()
        s_model.train()
        
        t_model.to(self.device)
        generator.to(self.device)
        s_model.to(self.device)
        
        for i in range(n_iters):
            s_optimizer.zero_grad()
        
            if direction == 'sl':
                s_policy = self.eq_policy[self.eq_depths[self.eq_depths.index(t_eq)+1]]
                
                begin_exit = len(t_model.config.exits)-2 if len(t_model.config.exits) > 1 else None
                
                # t_logits = t_model(latent=gen_latent, exit_idxs=(begin_exit, len(t_model.config.exits)-1))
                # s_logits = s_model(latent=gen_latent, exit_idxs=(begin_exit, s_exit))
                
                t_logits = t_policy.sf(t_model(gen_latent, is_latent=self.is_latent))
                s_logits = s_policy.sf(s_model(gen_latent, stop_exit=g_exit, is_latent=self.is_latent))
                
                dist_loss = self.kd_dist_ratio*self.dist_criterion(s_logits, t_logits)
                angle_loss = self.kd_angle_ratio*self.angle_criterion(s_logits, t_logits)
                dark_loss = self.kd_dark_ratio*self.dark_criterion(s_logits, t_logits)
                
                DIST_LOSS += dist_loss
                ANGLE_LOSS += angle_loss
                DARK_LOSS += dark_loss
                
                loss = dist_loss + angle_loss + dark_loss
            
            elif direction == 'ls':
                t_logits = t_policy.sf(t_model(gen_latent, stop_exit=g_exit, is_latent=self.is_latent))
                s_logits = s_policy.sf(s_model(gen_latent, is_latent=self.is_latent))
                
                kd_loss = self.kd_criterion(s_logits, t_logits)
                
                KD_LOSS += kd_loss
                
                loss = kd_loss
            
            loss.backward()
            s_optimizer.step()
        if direction == 'sl':
            print(f'dist_loss:{DIST_LOSS/n_iters}, angle_loss: {ANGLE_LOSS/n_iters}, dark_loss: {DARK_LOSS/n_iters}')
        else:
            print(f'kd_loss:{KD_LOSS/n_iters}')
        
    
    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self.eq_model[client.eq_depth])
    
    
    def save_model(self, model_save_path, generator_save_path):
        self.global_model.save_model(model_save_path)
        
        # TODO save sl & ls generators
        # generator_save_path = '.'.join(generator_save_path.split('.')[:-1])
        # for i, g in enumerate(self.generators):
        #     g_model = g[0]
        #     generator_save_path_i = f'{generator_save_path}_{i}.pth'
        #     g_model.save_model(generator_save_path_i)
        