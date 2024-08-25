import torch
import torch.nn as nn
import time
import random
import importlib
import copy

from typing import *

from utils.dataloader_utils import load_dataset_loader
from utils.dataprocess import DataProcessor

from utils.modelload.model import BaseModule
from utils.train_utils import AdamW

CLASSES = {'cifar100-224-d03': 100, 'sst2': 2}

class BaseClient:
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        self.id = id
        self.args = args
        self.dataset_train, self.loader_train = load_dataset_loader(args=args, file_name='train', id=id, need_process=False)
        self.dataset_valid, self.loader_valid = load_dataset_loader(args=args, file_name='valid', id=id, need_process=False)
        self.device = args.device
        self.exits = exits
        self.server = None

        self.y_distribute = [0 for _ in range(CLASSES[args.dataset])]
        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.eq_depth = depth
        self.model = model.to(self.device)
        self.exits_num = len(self.exits)
        
        self.loss_func = nn.CrossEntropyLoss()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        self.optim = torch.optim.Adam(params=optimizer_grouped_parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
        # self.optim = torch.optim.SGD(params=self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim, gamma=args.gamma)

        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
        }

        self.training_time = None
        self.lag_level = args.lag_level
        self.weight = 1
        self.submodel_weights = {}
        
        # == policy ==
        args.exits_num = self.exits_num
        policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
        self.policy = policy_module.Policy(args)
            
        # == y_distribute ==
        for idx, data in enumerate(self.loader_train):
            labels = data['labels'].cpu().tolist()
            for y in labels:
                self.y_distribute[y] += 1
                        
        
    def run(self):
        raise NotImplementedError()


    def train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()
                batch = {}
                for key in data.keys():
                    batch[key] = data[key].to(self.device)
                label = batch['labels']

                ce_loss = torch.zeros(1).to(self.device)
                ce_loss = self.policy.train(self.model, batch, label.view(-1))
                ce_loss.backward()
                self.optim.step()
                batch_loss.append(ce_loss.detach().cpu().item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def clone_model(self, target):
        p_tensors = target.parameters_to_tensor(is_split=True)
        idx = self.args.eq_depths.index(self.eq_depth)
        self.model.tensor_to_parameters(torch.cat(p_tensors[:idx+1], 0))

    def local_valid(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_valid:
                batch = {}
                for key in data.keys():
                    batch[key] = data[key].to(self.device)
                labels = batch['labels'].view(-1)
                
                exits_logits = self.model(**batch)
                exits_logits = self.policy(exits_logits)
                
                for exit_logits in exits_logits:
                    _, predicted = torch.max(exit_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        self.scheduler.step()
        # self.optim = torch.optim.SGD(params=self.model.parameters(), lr=(self.lr * (self.args.gamma ** self.server.round)))


class BaseServer:
    def __init__(self, id, args, dataset, clients:List[BaseClient], eq_model=None, global_model=None, eq_exits=None):
        # super().__init__(id, args, dataset)
        self.args = args
        self.eq_exits = eq_exits
        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.clients = clients
        self.sampled_clients = []
        self.total_round = args.rnd
        self.device = args.device
        self.eq_model:Dict[int:BaseModule] = {eq: model.to(self.device) for eq, model in eq_model.items()}
        self.global_model = global_model.to(self.device)
        self.eq_depths = list(self.eq_model.keys())
        self.sampled_submodel_clients: Dict[int:List[BaseClient]] = {}

        self.round = 0
        self.wall_clock_time = 0

        self.received_params = []

        for client in self.clients:
            client.server = self

        self.TO_LOG = True
            
        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
        }
        
        # == ratio of each classes for each eq ==  
        self.eq_y = {}
        for client in self.clients:
            self.eq_y.setdefault(client.eq_depth, []).append(client.y_distribute)
        for eq_depth in self.eq_depths:
            y_distribute = [sum(column) for column in zip(*self.eq_y[eq_depth])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            self.eq_y[eq_depth] = y_distribute
            
        self.crt_epoch = 0
        
        # == eq_policy ==
        self.eq_policy = {}
        for eq_depth in self.eq_depths:
            args.exits_num = self.eq_exits[eq_depth]
            policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
            policy = policy_module.Policy(args)
            self.eq_policy[eq_depth] = policy
        

    def run(self):
        raise NotImplementedError()

    def sample(self):
        self.sampled_submodel_clients.clear()
        
        sample_num = int(self.sample_rate * self.client_num)
        
        check_all_depths_sampled = {}
        while sum(check_all_depths_sampled.values()) != len(self.eq_depths):
            check_all_depths_sampled.clear()
            self.sampled_clients: List[BaseClient] = random.sample(self.clients, sample_num)
            for client in self.sampled_clients:
                check_all_depths_sampled[client.eq_depth] = 1

        for submodel_depth in self.eq_depths:
            for client in self.sampled_clients:
                if client.eq_depth >= submodel_depth:
                    self.sampled_submodel_clients.setdefault(submodel_depth, []).append(client)

        for submodel_depth, clients in self.sampled_submodel_clients.items():
            total_samples = sum(len(client.dataset_train) for client in clients)
            for client in clients:
                client.submodel_weights[submodel_depth] = len(client.dataset_train) / total_samples
        
    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        for client in self.sampled_clients:
            # client.clone_model(self.eq_model[client.eq_depth])
            client.clone_model(self.global_model)

    def client_update(self):
        for client in self.sampled_clients:
            client.model.train()
            client.reset_optimizer()
            start_time = time.time()
            client.run()
            end_time = time.time()
            client.training_time = (end_time - start_time) * client.lag_level
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = ()
        for idx, submodel_depth in enumerate(self.eq_depths):
            self.received_params += ([client.model.parameters_to_tensor(is_split=True)[idx] * client.submodel_weights[submodel_depth]
                                for client in self.sampled_submodel_clients[submodel_depth]],)

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        avg_tensor = torch.cat(avg_eq_tensor, 0)
        self.global_model.tensor_to_parameters(avg_tensor)

    def valid_all(self):
        for client in self.clients:
            c_metric = client.metric
            if client in self.sampled_clients:
                self.metric['loss'].append(c_metric['loss'].last())

            client.clone_model(self.global_model)
            client.local_valid()

            self.metric['acc'].append(c_metric['acc'].last())
        return self.analyse_metric()

    def analyse_metric(self):
        acc = self.metric['acc'].avg()
        loss = self.metric['loss'].avg()
        std = self.metric['acc'].std()

        self.metric['acc'].clear()
        self.metric['loss'].clear()

        return {'loss': loss,
                'acc': acc,
                'std': std}