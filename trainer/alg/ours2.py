import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW



def add_args(parser):
    parser.add_argument('--is_latent', default=False, type=bool)
    parser.add_argument('--is_feature', default=False, type=bool)
    
    parser.add_argument('--s_epoches', default=5, type=int)
    
    parser.add_argument('--kd_gap', default=1, type=int)
    parser.add_argument('--kd_begin', default=0, type=int)
    parser.add_argument('--kd_lr', default=5e-2, type=float)
    parser.add_argument('--kd_response_ratio', default=3, type=float)
    parser.add_argument('--kd_dist_ratio', default=1, type=float)
    parser.add_argument('--kd_angle_ratio', default=2, type=float)
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
        self.aggregate()
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
            
            self.train_mean = torch.mean(self.clients_embeddings, dim=0)
            self.train_std = None
            # self.train_mean = self.clients_embeddings.mean([0,2], keepdim=True)
            # self.train_std = self.clients_embeddings.std([0,2], keepdim=True)
            del self.clients_embeddings
            # print(self.train_mean, self.train_mean.shape)
            # print(self.train_std, self.train_std.shape)
   
   
    def lr_scheduler(self,):
        # == decay lr for generator & global model ==
        for g in self.generators:
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
        # param_optimizer = list(self.global_model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #     'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        # ]
        # optimizer = torch.optim.Adam(params=self.global_model.parameters(), lr=self.kd_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=False)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
        self.global_optimizer = torch.optim.SGD(params=self.global_model.parameters(), lr=self.kd_lr)
        self.global_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.global_optimizer, gamma=self.gamma)
        
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
            generator = Generator_CIFAR(args) if self.is_latent is False else Generator_LATENT(args)
            optimizer = torch.optim.Adam(params=generator.parameters(), lr=self.g_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
            self.generators.append([generator, optimizer, lr_scheduler])
     
     
    def get_batch(self, gen_latent, y_input):
        batch = {}
        if 'cifar' in self.args.dataset:
            batch['pixel_values'] = gen_latent
        else:
            batch['input_ids'] = gen_latent
            batch['attention_mask'] = y_input[1]
        return batch
     
     
    def get_gen_latent(self, ):
        for g in self.generators:
            g[0].to(self.device)
            g[0].train()
        y_input_g = {}
        eps_g = {}
        for t_exit in range(len(self.eq_exits[max(self.eq_depths)])):
            # == new y based y_distribute ==
            attend_eq = [eq_depth for eq_depth in self.eq_depths if t_exit < len(self.eq_exits[eq_depth])]
            y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            
            # TODO two classes for GLUE
            y_sl_distribute = {y:[sum(column) for column in zip(*[[sl*self.eq_num[eq] for sl in self.eq_y_sl[eq][y]] for eq in attend_eq])] for y in range(0,2)}
            y_sl_distribute = {y: [sl/sum(y_sl_distribute[y]) for sl in y_sl_distribute[y]] for y in range(0, 2)}
            
            y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            if self.args.dataset in GLUE:
                attention_mask = ()
                for i in range(self.args.bs):
                    y = y_input.cpu().tolist()[i]
                    sentence_len = torch.tensor(random.choices(range(len(y_sl_distribute[y])), weights=y_sl_distribute[y], k=1), dtype=torch.long)
                    mask = torch.zeros(128)
                    mask[:sentence_len] = 1
                    attention_mask += (mask.to(self.device), )
                attention_mask = torch.stack(attention_mask)
                y_input_g[t_exit] = (y_input, attention_mask)
            else:
                y_input_g[t_exit] = (y_input, )
            # == data ==
            # gen_latent, eps = self.generators[t_exit][0](y_input, )
            
            # gen_latent_g[t_exit] = gen_latent
            g = self.generators[t_exit][0]
            eps_g[t_exit] = torch.rand((y_input.shape[0], g.noise_dim)).to(self.device)

            # attend_logits = ()
            # for eq_depth in attend_eq:
            #     attend_logits += (self.eq_policy[eq_depth].sf(self.eq_model[eq_depth](gen_latent, stop_exit=t_exit, is_latent=self.is_latent)) * self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in attend_eq]), )
            # attend_logits = sum(attend_logits)
            
            # ts_logits[t_exit] = attend_logits
        return y_input_g, eps_g
    
    
    def finetune(self):
        # == train generator & global model ==
        for _ in range(self.s_epoches):
            y_input_g, eps_g = self.get_gen_latent()
            gen_latent_g = {}
            for t_exit in range(len(self.eq_exits[max(self.eq_depths)])):
                gen_latent = self.generators[t_exit][0](y_input_g[t_exit], eps_g[t_exit])
                gen_latent_g[t_exit] = gen_latent      
            if self.crt_epoch % self.g_gap == 0 and self.crt_epoch >= self.g_begin:
                self.train_generators(y_input_g, gen_latent_g, eps_g)
            
            for exit, y_input in y_input_g.items():
                if isinstance(y_input, tuple): y_input_g[exit] = tuple(tensor.detach() for tensor in y_input)
            
            gen_latent_g = {}
            for t_exit in range(len(self.eq_exits[max(self.eq_depths)])):
                gen_latent = self.generators[t_exit][0](y_input_g[t_exit], eps_g[t_exit])
                gen_latent_g[t_exit] = gen_latent.detach()
            if self.crt_epoch % self.kd_gap == 0 and self.crt_epoch >= self.kd_begin:
                self.finetune_global_model(y_input_g, gen_latent_g)
                
    
    def train_generators(self, y_input_g, gen_latent_g, eps_g):
        for g in self.generators:
            g[0].to(self.device)
            g[0].train()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        
        # == train generators ==
        for exit_idx, g in enumerate(self.generators):
            print(f'============{exit_idx}============')
            self.update_generator(g, exit_idx, self.g_n_iters, y_input_g, gen_latent_g, eps_g)
    
    
    def update_generator(self, g, exit_idx, n_iters, y_input_g, gen_latent_g, eps_g):
        
        CE_LOSS, DIV_LOSS, GAP_LOSS, STT_LOSS = 0, 0, 0, 0
        
        generator = g[0]
        optimizer = g[1]
        
        attend_eq = [eq_depth for eq_depth in self.eq_depths if exit_idx < len(self.eq_exits[eq_depth])]
        
        for i in range(n_iters):
            optimizer.zero_grad()
            
            y_input, gen_latent, eps = y_input_g[exit_idx], gen_latent_g[exit_idx], eps_g[exit_idx]
            # == div kd ce loss for G ==
            # == div loss for G ==
            div_loss = self.g_beta*generator.diversity_loss(eps, gen_latent)

            stt_loss = self.g_gamma*generator.statistic_loss(gen_latent, self.train_mean, self.train_std)
            # stt_loss = 0
            
            # == ensemble logits for attend eq's
            attend_logits = ()
            attend_feature = ()
            for eq_depth in attend_eq:
                r = self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in attend_eq])
                exits_logits, exits_feature = self.eq_model[eq_depth](**self.get_batch(gen_latent, y_input), stop_exit=exit_idx, is_latent=self.is_latent, rt_feature=True)
                
                attend_logits += (self.eq_policy[eq_depth].sf(exits_logits) * r, )
                attend_feature += (self.eq_policy[eq_depth].sf(exits_feature) * r, )
                
            attend_logits = sum(attend_logits)
            attend_feature = sum(attend_feature)
            
            ce_loss = self.g_alpha*self.ce_criterion(attend_logits, y_input[0].view(-1))
            gap_loss = self.g_eta*torch.zeros(1).to(self.device)
            
            if exit_idx != 0:
                former_attend_eq = [eq_depth for eq_depth in self.eq_depths if exit_idx-1 < len(self.eq_exits[eq_depth])]
                former_attend_logits = ()
                former_attend_feature = ()
                for eq_depth in former_attend_eq:
                    r = self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in former_attend_eq])
                    
                    exits_logits, exits_feature = self.eq_model[eq_depth](**self.get_batch(gen_latent, y_input), stop_exit=exit_idx-1, is_latent=self.is_latent, rt_feature=True)
                    former_attend_logits += (self.eq_policy[eq_depth].sf(exits_logits) * r, )
                    former_attend_feature += (self.eq_policy[eq_depth].sf(exits_feature) * r, )

                former_attend_logits = sum(former_attend_logits)
                former_attend_feature = sum(former_attend_feature)
                
                # relation_loss = self.kd_dist_ratio*self.dist_criterion(former_attend_feature.detach(), attend_feature) + self.kd_angle_ratio*self.angle_criterion(former_attend_feature.detach(), attend_feature) + self.kd_dark_ratio*self.dark_criterion(former_attend_feature.detach(), attend_feature) if self.is_feature else self.kd_dist_ratio*self.dist_criterion(former_attend_logits.detach(), attend_logits) + self.kd_angle_ratio*self.angle_criterion(former_attend_logits.detach(), attend_logits) + self.kd_dark_ratio*self.dark_criterion(former_attend_logits.detach(), attend_logits)
                kd_loss = self.g_eta*self.kd_criterion(former_attend_logits, attend_logits.detach())
                gap_loss = kd_loss
                
                
                # kd_loss = self.g_eta*torch.mean(torch.mean(torch.abs(former_attend_logits-attend_logits.detach()), dim=1))
            
            loss = ce_loss + div_loss - gap_loss + stt_loss
            loss.backward(retain_graph=True) if i < n_iters - 1 else loss.backward()
            
            CE_LOSS += ce_loss
            DIV_LOSS += div_loss
            GAP_LOSS += gap_loss
            STT_LOSS += stt_loss
            
            optimizer.step()
        print(f'ce_loss:{CE_LOSS/n_iters}, div_loss: {DIV_LOSS/n_iters}, gap_loss: {GAP_LOSS/n_iters}, stt_loss: {STT_LOSS/n_iters}')
    
        
    def finetune_global_model(self, y_input_g, gen_latent_g):
        # == finetune global model , multi teacher to teach each exit ==
        for g in self.generators:
            g[0].eval()
        for eq_model in self.eq_model.values():
            eq_model.to(self.device)
            eq_model.eval() 
        self.global_model.train()
        
        self.teach_global_model(self.generators, self.kd_n_iters, y_input_g, gen_latent_g)

    
    def teach_global_model(self, gs, n_iters, y_input_g, gen_latent_g):
        
        t_logits_g, t_feature_g = {}, {}
        
        for t_exit in range(len(self.eq_exits[max(self.eq_depths)])):
            # == new y based y_distribute ==
            attend_eq = [eq_depth for eq_depth in self.eq_depths if t_exit < len(self.eq_exits[eq_depth])]
            # y_distribute = [sum(column) for column in zip(*[[y*self.eq_num[eq] for y in self.eq_y[eq]] for eq in attend_eq])]
            # y_distribute = [y/sum(y_distribute) for y in y_distribute]
            # y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=self.args.bs), dtype=torch.long).to(self.device)
            
            # # == data ==
            # gen_latent, eps = gs[t_exit][0](y_input, )
            # gen_latents[t_exit] = gen_latent.detach()
            
            y_input, gen_latent = y_input_g[t_exit], gen_latent_g[t_exit]

            attend_logits = ()
            attend_feature = ()
            for eq_depth in attend_eq:
                r = self.eq_num[eq_depth] / sum([self.eq_num[eq_depth] for eq_depth in attend_eq])
                exits_logits, exits_feature = self.eq_model[eq_depth](**self.get_batch(gen_latent, y_input), stop_exit=t_exit, is_latent=self.is_latent, rt_feature=True)
                attend_logits += (self.eq_policy[eq_depth].sf(exits_logits) * r, )
                attend_feature += (self.eq_policy[eq_depth].sf(exits_feature) * r, )
            attend_logits = sum(attend_logits)
            attend_feature = sum(attend_feature)
            
            t_logits_g[t_exit] = attend_logits.detach()
            t_feature_g[t_exit] = attend_feature.detach()
            
        Losses = []
        for _ in range(n_iters):
            self.global_optimizer.zero_grad()
            Loss = torch.zeros(1).to(self.device)
            
            t_exit_s_exit = {}
            for s_exit in range(len(self.eq_exits[max(self.eq_depths)])):
                t_exits = (s_exit-1, s_exit, s_exit+1)
                for t_exit in t_exits:
                    if t_exit >= 0 and t_exit < len(self.eq_exits[max(self.eq_depths)]):
                        t_exit_s_exit.setdefault(t_exit, []).append(s_exit)
            
            t_exit_max_s_logits = {}
            t_exit_max_s_feature = {}
            for t_exit in t_exit_s_exit.keys():
                max_s_exit = max(t_exit_s_exit[t_exit])
                max_s_logits, max_s_feature = self.global_model(**self.get_batch(gen_latent_g[t_exit], y_input_g[t_exit]), stop_exit=max_s_exit, is_latent=self.is_latent, rt_feature=True)
                t_exit_max_s_logits[t_exit] = max_s_logits
                t_exit_max_s_feature[t_exit] = max_s_feature
            
            
            
            for s_exit in range(len(self.eq_exits[max(self.eq_depths)])):
                loss = torch.zeros(1).to(self.device)
                t_exits = (s_exit-1, s_exit, s_exit+1)
                for t_exit in t_exits:
                    if t_exit >= 0 and t_exit < len(self.eq_exits[max(self.eq_depths)]):

                        s_logits = self.eq_policy[max(self.eq_depths)].sf(t_exit_max_s_logits[t_exit][:s_exit+1])
                        s_feature = self.eq_policy[max(self.eq_depths)].sf(t_exit_max_s_feature[t_exit][:s_exit+1])
                        
                        t_logits, t_feature = t_logits_g[t_exit], t_feature_g[t_exit]
                        
                        # loss += self.kd_response_ratio*self.kd_criterion(s_logits, t_logits)
                        if t_exit >= s_exit:
                            loss += self.kd_response_ratio*self.kd_criterion(s_logits, t_logits)
                        else:
                            loss += self.kd_dist_ratio*self.dist_criterion(s_feature, t_feature) + self.kd_angle_ratio*self.angle_criterion(s_feature, t_feature) + self.kd_dark_ratio*self.dark_criterion(s_feature, t_feature) if self.is_feature else self.kd_dist_ratio*self.dist_criterion(s_logits, t_logits) + self.kd_angle_ratio*self.angle_criterion(s_logits, t_logits) + self.kd_dark_ratio*self.dark_criterion(s_logits, t_logits)
                
                # Loss += loss
                Loss += loss * (s_exit+1) / (sum([i+1 for i in range(len(self.eq_exits[max(self.eq_depths)]))]))
  
            Loss.backward()
            self.global_optimizer.step()
            Losses.append(Loss.item())
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
        for i, g in enumerate(self.generators):
            g_model = g[0]
            generator_save_path_i = f'{generator_save_path}_{i}.pth'
            g_model.save_model(generator_save_path_i)
        