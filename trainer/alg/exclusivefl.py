import torch
import random

from typing import *

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    return parser.parse_args()

class Client(BaseClient):
    def run(self):
        self.train()
    
    # def train(self):
    #     # === train ===
    #     batch_loss = []
    #     for epoch in range(self.epoch):
    #         for idx, data in enumerate(self.loader_train):
    #             self.optim.zero_grad()
    #             batch = {}
    #             for key in data.keys():
    #                 batch[key] = data[key].to(self.device)
    #             label = batch['labels']
                
    #             loss = torch.zeros(1).to(self.device)
    #             exit_logits = self.model(**batch)
    #             for logits in exit_logits:
    #                 loss += self.loss_func(logits, label.view(-1))
    #             loss.backward()
    #             self.optim.step()
    #             batch_loss.append(loss.item())

    #     # === record loss ===
    #     self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
    
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None):
        super().__init__(id, args, dataset, clients, eq_model, global_model)
        self.eq_depths = [12]
        largest_num = int(args.eq_ratios[-1] * len(self.clients))
        sample_rate = self.sample_rate / (args.eq_ratios[-1])
        self.sample_rate = sample_rate if sample_rate < 1 else 1
        self.clients = self.clients[-largest_num:]
        self.client_num = len(self.clients)

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        exclusive_eq_depth = self.eq_depths[-1]
        self.received_params = ([client.model.parameters_to_tensor() * client.submodel_weights[exclusive_eq_depth]
                                for client in self.sampled_submodel_clients[exclusive_eq_depth]],)
