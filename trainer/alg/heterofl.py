import torch

from trainer.baseHFL import BaseServer, BaseClient
from utils.train_utils import crop_tensor_dimensions, aggregate_scale_tensors
from typing import *

def add_args(parser):
    return parser

class Client(BaseClient):
    
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        super().__init__(id, args, dataset, model, depth, exits)
        
        depth = 12
        self.width_scale = self.eq_depth/depth
        origin_hidden_size = args.origin_width[0]
        origin_intermediate_size = args.origin_width[1]
        self.origin_target = {origin_hidden_size: self.model.config.hidden_size, origin_intermediate_size: self.model.config.intermediate_size}
        
    
    def run(self):
        self.train()
        
    def clone_model(self, target):
        target_state_dict = target.state_dict()
        
        new_state_dict = {}
        for name, param in self.model.named_parameters():
            if target_state_dict[name].shape != param.shape:
                prune_param = crop_tensor_dimensions(target_state_dict[name], self.origin_target)
            else: prune_param = target_state_dict[name]
            new_state_dict[name] = prune_param
        self.model.load_state_dict(new_state_dict)
        

class Server(BaseServer):
    def run(self):
        self.sample()
        # print('sample')
        self.downlink()
        # print('downlink')
        self.client_update()
        # print('client_update')
        self.uplink()
        # print('unlink')
        self.aggregate()
        # print('aggregate')
    
    def weighted(self, named_grads, weight):
        for name, grad in named_grads.items():
            grad *= weight
        return named_grads
    
    
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = ()
        sum_ = sum([len(client.dataset_train) for client in self.sampled_clients])
        
        for client in self.clients:
            self.received_params += ({'state_dict': client.model.state_dict(), 'sample': len(client.dataset_train)},)
    
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        
        state_dict_list = [dct['state_dict'] for dct in self.received_params]
        sample_list = [dct['sample'] for dct in self.received_params]
        
        aggregated_state_dict = {}
        
        name_params = {}
        for state_dict in state_dict_list:
            for name, param in state_dict.items():
                name_params.setdefault(name, []).append(param)
        
        for name, params in name_params.items():
            aggregated_state_dict[name] = aggregate_scale_tensors(params, sample_list, self.device)
        
        self.global_model.load_state_dict(aggregated_state_dict)
