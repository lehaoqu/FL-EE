import torch
import torch.nn as nn

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    parser.add_argument('--T', type=float, default=1, help="kd T")
    return parser

class Client(BaseClient):
    
    def __init__(self, id, args, dataset, model=None, depth=None, exits=None):
        super().__init__(id, args, dataset, model, depth, exits)
        self.T = args.T
        
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
                    ce_slim_ratios = slim_ratios if self.args.slim_ce else [1.0]

                    ce_loss = torch.zeros(1).to(self.device)
                    reefl_kd_loss = torch.zeros(1).to(self.device)
                    ratio_exits_logits = {}

                    for slim_ratio in slim_ratios:
                        set_width_ratio(slim_ratio, self.model)

                        if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                            self.policy.train_meta(self.model, batch, label, self.optim)

                        exits_ce_loss, exits_logits = self.policy.train(self.model, batch, label)
                        ce_loss += sum(exits_ce_loss) / len(ce_slim_ratios) if slim_ratio in ce_slim_ratios else 0.0
                        ratio_exits_logits[slim_ratio] = exits_logits
                        teacher_index = torch.argmin(torch.stack(exits_ce_loss))
                        teacher_logits = exits_logits[teacher_index]

                        # Original ReeFL KD across exits (within the same width)
                        for i, student_logits in enumerate(exits_logits):
                            if i == teacher_index: continue
                            else: 
                                reefl_kd_loss += kd_loss_func(student_logits, teacher_logits.detach())


                    # Slimmable KD across widths: distill smaller widths from width=1.0 at each exit
                    t_exits_logits = ratio_exits_logits.get(1.0)
                    slim_kd_loss = torch.zeros(1).to(self.device)
                    if self.args.slim_kd:
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

                    loss = ce_loss + reefl_kd_loss + slim_kd_loss
                    loss.backward()
                    self.optim.step()
                    batch_loss.append(loss.detach().cpu().item())
                    set_width_ratio(1.0, self.model)

                else:
                    if self.policy.name == 'l2w' and idx % self.args.meta_gap == 0:
                        self.policy.train_meta(self.model, batch, label, self.optim)

                    exits_ce_loss, exits_logits = self.policy.train(self.model, batch, label)
                    ce_loss = sum(exits_ce_loss)
                    
                    teacher_index = torch.argmin(torch.stack(exits_ce_loss))
                    teacher_logits = exits_logits[teacher_index]
                    
                    kd_loss = torch.zeros(1).to(self.device)
                    for i, student_logits in enumerate(exits_logits):
                        if i == teacher_index: continue
                        else: 
                            kd_loss += kd_loss_func(student_logits, teacher_logits.detach())
                    loss = ce_loss + kd_loss
                    loss.backward()
                    self.optim.step()
                    batch_loss.append(loss.detach().cpu().item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


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
        