import torch
import torch.nn as nn
import time
import random
import importlib
import copy

from typing import *

from utils.dataloader_utils import load_dataset_loader
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset
from dataset.imagenet_dataset import TinyImageNetClassificationDataset
from dataset.speechcmd_dataset import SPEEDCMDSClassificationDataset
from utils.dataprocess import DataProcessor
from utils.train_utils import crop_tensor_dimensions, aggregate_scale_tensors, kd_loss_func

from utils.modelload.model import BaseModule
from utils.train_utils import AdamW

CLASSES = {'speechcmds':35, 'imagenet':200, 'svhn':10, 'cifar100_noniid1000': 100, 'cifar100_noniid1': 100, 'cifar100_noniid0.1': 100, 'cifar100-224-d03-1200': 100, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'qnli': 2, 'rte': 2, 'wnli': 2}
GLUE = {'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli'}

def add_args(parser):
    parser.add_argument('--T_slim', type=float, default=3.0, help="kd T for slim")
    return parser

class BaseClient:
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        self.id = id
        self.args = args
        self.dataset_train, self.loader_train = load_dataset_loader(args=args, file_name='train', id=id)
        self.dataset_valid, self.loader_valid = load_dataset_loader(args=args, file_name='valid', id=id)
        self.device = args.device
        self.exits = exits
        self.server = None

        self.y_distribute = [0 for _ in range(CLASSES[args.dataset])]
        # == label -> sentence len for GLUE ==
        self.y_sl = {0: [0 for _ in range(128)], 1: [0 for _ in range(128)]}
        self.lr = args.lr
        self.batch_size = args.bs
        self.epoch = args.epoch
        self.eq_depth = depth
        self.model = model
        self.model_arch = copy.deepcopy(model)
        self.exits_num = len(self.exits)
        
        self.loss_func = nn.CrossEntropyLoss()
        if args.optim == 'adam':
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            self.optim = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-08)
        else:   
            self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optim, gamma=args.gamma)

        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
        }

        self.training_time = None
        self.lag_level = 1
        self.weight = 1
        self.submodel_weights = {}
        
        # == policy ==
        args.exits_num = self.exits_num
        policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
        self.policy = policy_module.Policy(args)
            
        # == y_distribute ==
        if args.dataset not in GLUE:
            for idx, data in enumerate(self.loader_train):
                labels = data['labels'].view(-1).cpu().tolist()
                for y in labels:
                    self.y_distribute[y] += 1
        else:
            for idx, data in enumerate(self.loader_train):
                labels = data['labels'].view(-1).cpu().tolist()
                attention_mask = data['attention_mask'].cpu().tolist()
                for i in range(len(labels)):
                    label = labels[i]
                    sentence_len = len([x for x in attention_mask[i] if x != 0]) -1
                    self.y_distribute[label] += 1
                    self.y_sl[label][sentence_len] += 1
        # print(self.y_sl)
        
        # TODO == the max exit num is 4 ==
        self.origin_target_policy = {4: self.exits_num}
                        
        
    def run(self):
        raise NotImplementedError()


    def adapt_batch(self, data):
        batch = {}
        for key in data.keys():
            batch[key] = data[key].to(self.device)
            if key == 'pixel_values':
                if 'cifar' in self.args.dataset:
                    batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
                elif 'imagenet' in self.args.dataset:
                    batch[key] = TinyImageNetClassificationDataset.transform_for_vit(batch[key])
                elif 'speechcmds' in self.args.dataset:
                    batch[key] = SPEEDCMDSClassificationDataset.transform_for_vit(batch[key])
                else:
                    batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
        label = batch['labels'].view(-1)
        return batch, label

    def train(self):
        # === train ===
        self.model.to(self.device)
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()

                batch, label = self.adapt_batch(data)

                if self.args.slimmable:
                    ce_loss = torch.zeros(1).to(self.device)
                    ratio_exits_logits = {}
                    ratio_exits_ce_loss = {}
                    for slim_ratio in self.args.slim_ratios:
                        from utils.modelload.slimmable import set_width_ratio
                        set_width_ratio(slim_ratio, self.model)
                        if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                            self.policy.train_meta(self.model, batch, label, self.optim)

                        exits_ce_loss, exits_logits = self.policy.train(self.model, batch, label)
                        ce_loss += sum(exits_ce_loss) / len(self.args.slim_ratios)
                        ratio_exits_ce_loss[slim_ratio] = exits_ce_loss
                        ratio_exits_logits[slim_ratio] = exits_logits
                        # print(f'Client {self.id} Slim {slim_ratio} is ok')

                    t_exits_logits = ratio_exits_logits[1.0]
                    kd_loss = torch.zeros(1).to(self.device)
                    for slim_ratio in self.args.slim_ratios:
                        if slim_ratio == 1.0:
                            continue
                        for idx, student_logits in enumerate(ratio_exits_logits[slim_ratio]):
                            teacher_logits = t_exits_logits[idx].detach()
                            kd_loss += kd_loss_func(student_logits, teacher_logits, T=self.args.T_slim) / (len(self.args.slim_ratios)-1)

                    loss = ce_loss + kd_loss
                    loss.backward()
                    self.optim.step()
                    batch_loss.append(ce_loss.detach().cpu().item())
                    set_width_ratio(1.0, self.model)

                else:
                    if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                        self.policy.train_meta(self.model, batch, label, self.optim)

                    exits_ce_loss, exits_logits = self.policy.train(self.model, batch, label)
                    ce_loss = sum(exits_ce_loss)
                    ce_loss.backward()
                    self.optim.step()
                    batch_loss.append(ce_loss.detach().cpu().item())
        
        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))
   
   
    def clearGPU(self):
        del self.model
        del self.optim
        torch.cuda.empty_cache()
        

    def clone_model(self, target):
        p_tensors = target.parameters_to_tensor(is_split=True)
        idx = self.args.eq_depths.index(self.eq_depth)
        self.model.tensor_to_parameters(torch.cat(p_tensors[:idx+1], 0))
        
    
    def clone_policy(self, target):
        if self.policy.name == 'l2w':
            target_state_dict = target.meta_net.state_dict()
            
            new_state_dict = {}
            for name, param in self.policy.meta_net.named_parameters():
                if target_state_dict[name].shape != param.shape:
                    prune_param = crop_tensor_dimensions(target_state_dict[name], self.origin_target_policy)
                else: prune_param = target_state_dict[name]
                new_state_dict[name] = prune_param
            self.policy.meta_net.load_state_dict(new_state_dict)
    

    def local_valid(self):
        self.model.eval()
        correct = 0
        total = 0
        corrects = [0 for _ in range(self.exits_num)]

        with torch.no_grad():
            for data in self.loader_valid:
                batch, labels = self.adapt_batch(data)
                
                exits_logits = self.model(**batch)
                exits_logits = self.policy(exits_logits)
                
                for i, exit_logits in enumerate(exits_logits):
                    _, predicted = torch.max(exit_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    corrects[i] += (predicted == labels).sum().item()
        acc = 100.00 * correct / total
        acc_exits = [100 * c / (total/self.exits_num) for c in corrects]
        self.metric['acc_exits'] = acc_exits
        self.metric['acc'].append(acc)

    def reset_optimizer(self, decay=True):
        if not decay:
            return
        # self.scheduler.step()
        if self.args.optim == 'adam':
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            self.optim = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=(self.lr * (self.args.gamma ** self.server.round)), betas=(0.9, 0.999), eps=1e-08)
        else:   
            self.optim = torch.optim.SGD(params=self.model.parameters(), lr=(self.lr * (self.args.gamma ** self.server.round)), momentum=0.9, weight_decay=1e-4)
        
        # self.optim = torch.optim.SGD(params=self.model.parameters(), lr=(self.lr * (self.args.gamma ** self.server.round)), momentum=0.9, weight_decay=1e-4)


class BaseServer:
    def __init__(self, id, args, dataset, clients:List[BaseClient], eq_model=None, global_model=None, eq_exits=None):
        # super().__init__(id, args, dataset)
        self.args = args
        self.valid_dataset, self.valid_dataloader = load_dataset_loader(args=args, eval_valids=True)
        self.test_dataset,  self.test_dataloader  = load_dataset_loader(args=args, file_name='test', shuffle=False)
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
        self.max_depth = max(self.eq_depths)
        self.max_exit = len(self.eq_exits[self.max_depth])

        self.round = 0
        self.wall_clock_time = 0

        self.received_params = []

        for client in self.clients:
            client.server = self

        self.TO_LOG = True
            
        self.metric = {
            'acc': DataProcessor(),
            'loss': DataProcessor(),
            'acc_exits': []
        }
        
        # == ratio of each classes for each eq ==  
        self.eq_y = {}
        self.eq_y_sl_t = {}
        self.eq_y_sl = {eq:{} for eq in self.eq_depths}
        for client in self.clients:
            self.eq_y.setdefault(client.eq_depth, []).append(client.y_distribute)
            self.eq_y_sl_t.setdefault(client.eq_depth, []).append(client.y_sl)

        for eq in self.eq_y_sl.keys():
            for y in range(0,2):
                self.eq_y_sl[eq][y] = [sum(column) for column in zip(*[self.eq_y_sl_t[eq][i][y] for i in range(len(self.eq_y_sl_t[eq]))])]
        
        for eq_depth in self.eq_depths:
            y_distribute = [sum(column) for column in zip(*self.eq_y[eq_depth])]
            y_distribute = [y/sum(y_distribute) for y in y_distribute]
            self.eq_y[eq_depth] = y_distribute
        
        if self.args.dataset in GLUE:
            for eq_depth in self.eq_depths:
                tmp = {}
                for label in range(0,2):
                    sl_distribute = self.eq_y_sl[eq_depth][label]
                    sl_distribute = [sl/sum(sl_distribute) for sl in sl_distribute]
                    tmp[label] = sl_distribute
                self.eq_y_sl[eq_depth] = tmp
        # print(self.eq_y_sl)
            
        self.crt_epoch = 0
        
        # == eq_policy ==
        self.eq_policy = {}
        for eq_depth in self.eq_depths:
            args.exits_num = len(self.eq_exits[eq_depth])
            policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
            policy = policy_module.Policy(args)
            self.eq_policy[eq_depth] = policy
    
    def adapt_batch(self, data):
        batch = {}
        for key in data.keys():
            batch[key] = data[key].to(self.device)
            if key == 'pixel_values':
                if 'cifar' in self.args.dataset:
                    batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
                elif 'imagenet' in self.args.dataset:
                    batch[key] = TinyImageNetClassificationDataset.transform_for_vit(batch[key])
                elif 'speechcmds' in self.args.dataset:
                    batch[key] = SPEEDCMDSClassificationDataset.transform_for_vit(batch[key])
                else:
                    batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
        label = batch['labels'].view(-1)
        return batch, label

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

        for client in self.sampled_clients:
            client.model = client.model_arch
            client.reset_optimizer()

        for submodel_depth in self.eq_depths:
            for client in self.sampled_clients:
                if client.eq_depth >= submodel_depth:
                    self.sampled_submodel_clients.setdefault(submodel_depth, []).append(client)

        for submodel_depth, clients in self.sampled_submodel_clients.items():
            total_samples = sum(len(client.dataset_train) for client in clients)
            for client in clients:
                client.submodel_weights[submodel_depth] = len(client.dataset_train) / total_samples
                
        self.sampled_eq_clients = {}
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
            # client.clone_model(self.eq_model[client.eq_depth])
            client.clone_model(self.global_model)
            client.clone_policy(self.eq_policy[max(self.eq_depths)])
            

    def client_update(self):
        for client in self.sampled_clients:
            client.model.train()
            # client.reset_optimizer()
            start_time = time.time()
            client.run()
            end_time = time.time()
            client.training_time = (end_time - start_time) * 1
        self.wall_clock_time += max([client.training_time for client in self.sampled_clients])

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = ()
        for idx, submodel_depth in enumerate(self.eq_depths):
            self.received_params += ([client.model.parameters_to_tensor(is_split=True)[idx] * client.submodel_weights[submodel_depth]
                                for client in self.sampled_submodel_clients[submodel_depth]],)
        self.uplink_policy()
        for client in self.sampled_clients:
            client.clearGPU()
              
    def uplink_policy(self):
        if self.eq_policy[max(self.eq_depths)].name == 'l2w':
            self.received_params_policy = ()
            for client in self.clients:
                self.received_params_policy += ({'state_dict': client.policy.meta_net.state_dict(), 'sample': len(client.dataset_train)},)
          

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        avg_tensor = torch.cat(avg_eq_tensor, 0)
        self.global_model.tensor_to_parameters(avg_tensor)
        
        self.aggregate_policy()


    def aggregate_policy(self):
        if self.eq_policy[max(self.eq_depths)].name == 'l2w':
            state_dict_list = [dct['state_dict'] for dct in self.received_params_policy]
            sample_list = [dct['sample'] for dct in self.received_params_policy]
            
            aggregated_state_dict = {}
            
            name_params = {}
            for state_dict in state_dict_list:
                for name, param in state_dict.items():
                    name_params.setdefault(name, []).append(param)
            
            for name, params in name_params.items():
                aggregated_state_dict[name] = aggregate_scale_tensors(params, sample_list, self.device)
            
            self.eq_policy[max(self.eq_depths)].meta_net.load_state_dict(aggregated_state_dict)

    def valid_all(self):
        self.global_model.eval()
        correct = 0
        total = 0
        exit_num = len(self.eq_exits[max(self.eq_depths)])
        corrects = [0 for _ in range(exit_num)]

        with torch.no_grad():
            dataloader = self.test_dataloader if self.args.eval_test else self.valid_dataloader
            for data in dataloader:
            # for data in self.test_dataloader:
                batch, labels = self.adapt_batch(data)
                
                exits_logits = self.global_model(**batch)
                exits_logits = self.eq_policy[max(self.eq_depths)](exits_logits)
                
                for i, exit_logits in enumerate(exits_logits):
                    _, predicted = torch.max(exit_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    corrects[i] += (predicted == labels).sum().item()
        acc = 100.00 * correct / total
        acc_exits = [100 * c / (total/exit_num) for c in corrects]
        self.metric['acc'].append(acc)
        self.metric['acc_exits'].append(acc_exits)
        
        for client in self.clients:
            c_metric = client.metric
            if client in self.sampled_clients:
                self.metric['loss'].append(c_metric['loss'].last())

        return self.analyse_metric()

    def analyse_metric(self):
        acc = self.metric['acc'].avg()
        loss = self.metric['loss'].avg()
        std = self.metric['acc'].std()
        acc_exits = [sum(col) / len(col) for col in zip(*self.metric['acc_exits'])]

        self.metric['acc'].clear()
        self.metric['loss'].clear()
        self.metric['acc_exits'].clear()

        return {'loss': loss,
                'acc': acc,
                'std': std,
                'acc_exits': acc_exits}
        
    def save_model(self, model_save_path, generator_save_path):
        self.global_model.save_model(model_save_path)
 
        
    def calc_logits(self, best_model):
        dataloader = self.test_dataloader
        best_model.eval()
        all_sample_exits_logits = [[] for _ in range(self.max_exit)]
        all_sample_targets = []
        for i, data in enumerate(dataloader):           
            batch, y = self.adapt_batch(data)
            
            all_sample_targets.append(y)
            with torch.no_grad():
                exits_logits = self.eq_policy[self.max_depth](best_model(**batch))
                for i, exit_logits in enumerate(exits_logits):
                    # _t = self.softmax(exit_logits)
                    # all_sample_exits_logits[i].append(_t)
                    all_sample_exits_logits[i].append(exit_logits)
        
        for i in range(self.max_exit):
            all_sample_exits_logits[i] = torch.cat(all_sample_exits_logits[i], dim=0)
        
        size = (len(all_sample_exits_logits), all_sample_exits_logits[0].size(0), all_sample_exits_logits[0].size(1))
        ts_preds = torch.zeros(size=size).to(self.device)
        for i in range(len(all_sample_exits_logits)):
            ts_preds[i] = all_sample_exits_logits[i]
            
        all_sample_targets = torch.cat(all_sample_targets, dim=0)
        self.test_exits_preds, self.test_targets, self.test_all_sample_exits_logits = ts_preds, all_sample_targets, all_sample_exits_logits
        return 

    def anytime(self, output):
        crt_list = [0 for _ in range(self.max_exit)]
        for i in range(self.max_exit):
            _, predicted = torch.max(self.test_exits_preds[i], 1)
            crt_list[i] += (predicted == self.test_targets).sum().item()
        
        acc_list = [100 * crt_list[i] / self.test_targets.shape[0] for i in range(self.max_exit)]
        output.write('Anytime:\n{}, avg:{}\n'.format(acc_list, sum(acc_list) / len(acc_list)))
        output.flush()