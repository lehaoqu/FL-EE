import torch
import torch.nn as nn

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    return parser.parse_args()

class Client(BaseClient):
    def run(self):
        self.train()
    
    def train(self):
        
        def kd_loss_func(pred, teacher):
            kld_loss = nn.KLDivLoss(reduction='batchmean')
            log_softmax = nn.LogSoftmax(dim=-1)
            softmax = nn.Softmax(dim=1)
            T=3
            _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
            return _kld
        
        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                self.optim.zero_grad()
                batch = {}
                for key in data.keys():
                    batch[key] = data[key].to(self.device)
                label = batch['labels']
                
                # == ce loss ==
                ce_loss = torch.zeros(1).to(self.device)
                exit_logits = self.model(**batch)
                ce_loss = self.policy.train(self.model, batch, label.view(-1))

                # == kd loss ==    
                kd_loss = torch.zeros(1).to(self.device)
                for i, teacher_logits in enumerate(exit_logits):
                    for j, student_logits in enumerate(exit_logits):
                        if i == j: continue
                        else: 
                            kd_loss += kd_loss_func(student_logits, teacher_logits) / (len(exit_logits)-1)
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
        