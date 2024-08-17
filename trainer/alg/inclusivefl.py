import torch
import torch.nn as nn
import random
from typing import *

from trainer.baseHFL import BaseServer, BaseClient
from utils.modelload.model import BaseModule

def add_args(parser):
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args, dataset, model=None, depth=None):
        super().__init__(id, args, dataset, model, depth)
        self.optim = self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
    
    def run(self):
        self.train()
    
    def train(self):
        
        def kd_loss_func(pred, teacher):
            kld_loss = nn.KLDivLoss(reduction='batchmean')
            log_softmax = nn.LogSoftmax(dim=-1)
            softmax = nn.Softmax(dim=1)
            T=0.1
            _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
            return _kld
        
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                
                # == ce loss ==
                ce_loss = torch.zeros(1).to(self.device)
                exit_logits = self.model(image)
                for logits in exit_logits:
                    ce_loss += self.loss_func(logits, label)
                
                # == kd loss ==    

                kd_loss = torch.zeros(1).to(self.device)
                # for i, teacher_logits in enumerate(exit_logits):
                #     for j, student_logits in enumerate(exit_logits):
                #         if i == j: continue
                #         else: 
                #             kd_loss += kd_loss_func(student_logits, teacher_logits) / (len(exit_logits)-1)
                loss = ce_loss + kd_loss
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(BaseServer):
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model)
        
        # == ditillation beta ==
        self.beta = 0.2
        
        # == fedada ==
        self.m_t = {}
        self.v_t = {}
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.tau = 0.000001
        self.eta = 0.001
    
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        
    def weighted(self, named_grads, weight):
        for name, grad in named_grads.items():
            grad *= weight
        return named_grads
    
    def sum(self, named_grads_list):
        sum_named_grads: Dict[str, torch.tensor] = {}
        for named_grads in named_grads_list:
            for name, grad in named_grads:
                sum_named_grads[name] = sum_named_grads.get(name, 0.0) + grad
        return sum_named_grads
    
    # get grads from model state_dict
    def get_named_from_dict(self, origin_named_grads, layer_idx_range=None, include_IC=True):
        named_grads = {}
        for idx, (name, param) in enumerate(origin_named_grads.items()):
            if layer_idx_range is not None:
                if len(layer_idx_range) == 1:
                    if self.get_layer_idx(name) != layer_idx_range: continue
                else:
                    if self.get_layer_idx(name) not in tuple(range(layer_idx_range)): continue
            if 'classifier' in name & include_IC is False: continue
            named_grads[name] = param.grad.detach()
        return named_grads
    
    # modifier grads
    def modifier_grads(self, origin_named_grad:dict, new_named_grads:dict):
        for name, grad in new_named_grads:
            origin_named_grad[name] = grad.clone()
        return origin_named_grad
    
    # state_dict to tensor
    def state_dict_to_tensor(self, state_dict: dict):
        params = []
        for (name, param) in state_dict.items():
            params.append(param.view(-1))
        return torch.nan_to_num(torch.cat(params, 0), nan=0.0, posinf=0.0, neginf=0.0)
    
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
        
        self.larger_eq_total_num = {eq_depth: sum([self.eq_num[i] for i in self.eq_depths if i >= eq_depth]) for eq_depth in self.eq_depths}
    
    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            client.clone_model(self.eq_model[client.eq_depth])
    
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = {}
        for idx, eq_depth in enumerate(self.eq_depths):
            self.received_params[eq_depth] = [self.weighted(client.model.state_dict(), client.weight) for client in self.sampled_eq_clients[eq_depth]]
            
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        
        # == Homomorphic aggregate ==
        momentum_eq = {}
        eq_named_grads: Dict[Dict[str, torch.tensor]] = {}
        for eq_depth in reversed(self.eq_depths):
            eq_received_params = self.sum(self.received_params[eq_depth])
            
            # == deta of eq_params -> grads ==
            eq_named_grads[eq_depth] = {n: b-a for (n, a), (n,b) in zip(self.eq_model[eq_depth].state_dict().items(), eq_received_params.items())}
            
            eq_model = self.eq_model[eq_depth]
            eq_index = self.eq_depths.index(eq_depth)
            
            # == is not smallest eq, attain momentum ==
            if eq_depth != min(self.eq_depths):
                next_small_eq_depth = self.eq_depths[eq_index-1]
                weighted_deeper_grads = [self.weighted(self.get_named_from_dict(eq_named_grads[eq_depth], layer_idx_range=depth, include_IC=False), 1/(eq_depth-next_small_eq_depth+1)) for depth in range(next_small_eq_depth-1, eq_depth)]
                momentum_eq[eq_depth] = self.sum(weighted_deeper_grads)
            
            # == is not largest eq, learn momentum ==
            if eq_depth != max(self.eq_depths):
                eq_last_layer_grads = self.get_named_from_dict(eq_named_grads[eq_depth], layer_idx_range=eq_depth-1, include_IC=False)
                next_larger_eq_depth = self.eq_depths[eq_index+1]
                momentum = momentum_eq[next_larger_eq_depth]
                eq_last_layer_grads = self.sum([self.weighted(momentum, self.beta), self.weighted(eq_last_layer_grads, 1-self.beta)])
                eq_named_grads[eq_depth] = self.modifier_grads(eq_named_grads[eq_depth], eq_last_layer_grads)
            
            eq_tensor_origin = eq_model.parameters_to_tensor()
            grad = self.state_dict_to_tensor(eq_named_grads[eq_depth])
            
            # == fedavg ==
            eq_tensor_updated = eq_tensor_origin + grad
            
            # == fedadam ==
            self.m_t[eq_depth] = self.beta_1 * self.m_t.get(eq_depth, torch.zeros_like(grad)) + (1 - self.beta_1) * grad
            self.v_t[eq_depth] = self.beta_2 * self.v_t.get(eq_depth, torch.zeros_like(grad)) + (1 - self.beta_2) * grad * grad
            eq_tensor_updated = eq_tensor_origin + self.eta * self.m_t[eq_depth] / (torch.sqrt(self.v_t[eq_depth]) + self.tau)
            
            # == update ==
            eq_model.tensor_to_parameters(eq_tensor_updated)
        
        # == Heterogeneous aggregation ==
        depth_weighted_tensor:Dict[int:List[torch.tensor]] = {}
        eq_tensors = {}
        for eq_depth in self.eq_depths:
            eq_model = self.eq_model[eq_depth]
            tensors = eq_model.parameters_to_tensor(is_split=True, is_inclusivefl=True)
            eq_tensors[eq_depth] = tensors
            for idx in range(len(tensors)-1):
                depth_weighted_tensor.setdefault(idx, []).append(tensors[idx] * self.eq_num[eq_depth]/self.larger_eq_total_num[eq_depth])
        
        updated_tensors = [sum(tensors) for (idx, tensors) in depth_weighted_tensor.items()]
        for idx, eq_depth in enumerate(self.eq_depths):
            if eq_depth != max(self.eq_depths):
                updated_tensor = torch.cat(updated_tensors[:idx+1].append(eq_tensors[-1]), 0)
            else:
                updated_tensor = torch.cat(updated_tensors, 0)
            eq_model: BaseModule = self.eq_model[eq_depth]
            eq_model.tensor_to_parameters(updated_tensor)