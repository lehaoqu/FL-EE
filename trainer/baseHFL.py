import torch
import torch.nn as nn
import time
import random

from typing import *

from utils.data_utils import read_client_data
from utils.modelload.modelloader import load_model
from utils.dataprocess import DataProcessor
from torch.utils.data import DataLoader


class BaseClient:
    def __init__(self, id, args, dataset, model=None, depth=None):
        self.id = id
        self.args = args
        self.dataset_train = read_client_data(args.dataset, self.id, is_train=True)
        self.dataset_test = read_client_data(args.dataset, self.id, is_train=False)
        self.device = args.device
        
        self.server = None

        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.eq_depth = depth
        self.model = model.to(self.device)
        
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)

        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
        }

        if self.dataset_train is not None:
            self.loader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )
        if self.dataset_test is not None:
            self.loader_test = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )

        self.training_time = None
        self.lag_level = args.lag_level
        self.weight = 1
        self.submodel_weights = {}

    def run(self):
        raise NotImplementedError()

    def train(self):
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                predict_label = self.model(image)
                loss = self.loss_func(predict_label, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def clone_model(self, target):
        p_tensors = target.parameters_to_tensor(is_split=True)
        idx = self.args.eq_depths.index(self.eq_depth)
        self.model.tensor_to_parameters(torch.cat(p_tensors[:idx+1], 0))

    def local_test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.loader_test:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                last_logits = self.model(images)[-1]
                _, predicted = torch.max(last_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100.00 * correct / total

        self.metric['acc'].append(acc)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        self.optim = torch.optim.SGD(params=self.model.parameters(),
                                     lr=(self.lr * (self.args.gamma ** self.server.round)))


class BaseServer:
    def __init__(self, id, args, dataset, clients, eq_model=None, global_model=None):
        # super().__init__(id, args, dataset)
        self.client_num = args.total_num
        self.sample_rate = args.sr
        self.clients = clients
        self.sampled_clients = []
        self.total_round = args.rnd
        self.eq_model = {eq: model.to(args.device) for eq, model in eq_model.items()}
        self.global_model = global_model.to(args.device)
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

    def test_all(self):
        for client in self.clients:
            c_metric = client.metric
            if client in self.sampled_clients:
                self.metric['loss'].append(c_metric['loss'].last())

            client.clone_model(self.global_model)
            client.local_test()

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