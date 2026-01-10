import torch
import torch.nn as nn
import copy

from typing import *
from functools import reduce
from trainer.baseHFL import BaseServer, BaseClient
from utils.train_utils import crop_tensor_dimensions, aggregate_scale_tensors

def add_args(parser):
    parser.add_argument('--T', type=float, default=1, help="kd T")
    return parser

class Client(BaseClient):
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        super().__init__(id, args, dataset, model, depth, exits)
        # self.scale_width = args.scale_width
        self.T = self.args.T
        depth = min(12, self.eq_depth+1)
        self.width_scale = self.eq_depth / depth
        origin_hidden_size = args.origin_width[0]
        origin_intermediate_size = args.origin_width[1]
        self.origin_target = {origin_hidden_size: self.model.config.hidden_size, origin_intermediate_size: self.model.config.intermediate_size}
        self.bert_em_position = {origin_hidden_size: self.model.config.hidden_size}
        # print(f'scale: {scale}, width: {self.origin_target}')
    
    def run(self):
        self.train()
    
    def train(self):
        
        def kd_loss_func(pred, teacher, T=self.T):
            kld_loss = nn.KLDivLoss(reduction='batchmean')
            log_softmax = nn.LogSoftmax(dim=-1)
            softmax = nn.Softmax(dim=1)
            _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
            return _kld
        
        # === train ===
        self.model.to(self.device)
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()
                batch, label = self.adapt_batch(data)

                if getattr(self.args, 'slimmable', False):
                    from utils.modelload.slimmable import set_width_ratio

                    slim_ratios = getattr(self.args, 'slim_ratios', [1.0])
                    if 1.0 not in slim_ratios:
                        slim_ratios = list(slim_ratios) + [1.0]

                    ce_loss = torch.zeros(1).to(self.device)
                    exit_kd_loss = torch.zeros(1).to(self.device)
                    ratio_exits_logits = {}

                    for slim_ratio in slim_ratios:
                        set_width_ratio(slim_ratio, self.model)

                        if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                            self.policy.train_meta(self.model, batch, label, self.optim)

                        exits_ce_loss, exits_logits = self.policy.train(
                            self.model,
                            batch,
                            label.view(-1),
                            ws=[i + 1 for i in range(self.exits_num)],
                        )
                        ce_loss += sum(exits_ce_loss) / len(slim_ratios)
                        ratio_exits_logits[slim_ratio] = exits_logits

                        # Original ScaleFL exit-KD (within the same width)
                        kd_local = torch.zeros(1).to(self.device)
                        teacher_logits = exits_logits[-1]
                        teacher_idx = self.exits_num - 1
                        for student_idx, student_logits in enumerate(exits_logits):
                            if student_idx < teacher_idx:
                                kd_local += kd_loss_func(student_logits, teacher_logits.detach()) * (student_idx + 1)
                        exit_kd_loss += kd_local / len(slim_ratios)

                    # Slimmable KD across widths: distill smaller widths from width=1.0 at each exit
                    t_exits_logits = ratio_exits_logits.get(1.0)
                    slim_kd_loss = torch.zeros(1).to(self.device)
                    denom = max(1, (len(slim_ratios) - 1))
                    if t_exits_logits is not None:
                        for slim_ratio in slim_ratios:
                            if slim_ratio == 1.0:
                                continue
                            for exit_idx, student_logits in enumerate(ratio_exits_logits[slim_ratio]):
                                teacher_logits = t_exits_logits[exit_idx].detach()
                                slim_kd_loss += kd_loss_func(
                                    student_logits,
                                    teacher_logits,
                                    T=getattr(self.args, 'T_slim', 1.0),
                                ) / denom

                    loss = (ce_loss + exit_kd_loss) / (self.exits_num * (self.exits_num + 1)) + slim_kd_loss / len(t_exits_logits)
                    loss.backward()
                    self.optim.step()
                    batch_loss.append(loss.detach().cpu().item())
                    set_width_ratio(1.0, self.model)

                else:
                    if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                        self.policy.train_meta(self.model, batch, label, self.optim)

                    # == ce loss ==
                    exits_ce_loss, exits_logits = self.policy.train(
                        self.model,
                        batch,
                        label.view(-1),
                        ws=[i + 1 for i in range(self.exits_num)],
                    )
                    ce_loss = sum(exits_ce_loss)

                    # == kd loss ==
                    kd_loss = torch.zeros(1).to(self.device)
                    teacher_logits = exits_logits[-1]
                    teacher_idx = self.exits_num - 1
                    for student_idx, student_logits in enumerate(exits_logits):
                        if student_idx < teacher_idx:
                            kd_loss += kd_loss_func(student_logits, teacher_logits.detach()) * (student_idx + 1)

                    loss = (ce_loss + kd_loss) / (self.exits_num * (self.exits_num + 1))
                    loss.backward()
                    self.optim.step()
                    batch_loss.append(loss.detach().cpu().item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))

    def clone_model(self, target):
        target_state_dict = target.state_dict()
        
        new_state_dict = {}
        for name, param in self.model.named_parameters():
            if target_state_dict[name].shape != param.shape:
                if 'bert.embeddings.position' in name: prune_param = crop_tensor_dimensions(target_state_dict[name], self.bert_em_position)
                else: prune_param = crop_tensor_dimensions(target_state_dict[name], self.origin_target)
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
        # print('uplink')
        self.aggregate()
        # print('aggregate')
        
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        self.received_params = ()

        for idx, submodel_depth in enumerate(self.eq_depths):
            self.received_params += ([{'state_dict': client.model.split_state_dict(blocks=self.global_model.config.blocks, ft=self.args.ft)[idx], 'sample': len(client.dataset_train)} for client in self.sampled_submodel_clients[submodel_depth]],)
        
        self.uplink_policy()
        
            
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        
        aggregated_state_dict = {}
        
        # == vertical ==
        for idx, eq_depth in enumerate(self.eq_depths):
            
            state_dict_list = [dct['state_dict'] for dct in self.received_params[idx]]
            sample_list = [dct['sample'] for dct in self.received_params[idx]]
            
            name_tensors = {}
            for state_dict in state_dict_list:
                for name, param in state_dict.items():
                    name_tensors.setdefault(name, []).append(param)    
                    
            # == horizontal ==
            for name, tensors in name_tensors.items():
                aggregated_state_dict[name] = aggregate_scale_tensors(tensors, sample_list, self.device)
                
        self.global_model.load_state_dict(aggregated_state_dict)

        self.aggregate_policy()