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
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                
                # == ce loss ==
                ce_loss = torch.zeros(1).to(self.device)
                exit_logits = self.model(image)
                exit_num = len(exit_logits)
                
                for i, logits in enumerate(exit_logits):
                    ce_loss += self.loss_func(logits, label) * (i+1)
                
                # == kd loss ==    
                kd_loss = torch.zeros(1).to(self.device)
                teacher_logits = exit_logits[-1]
                teacher_idx = exit_num-1
                for student_idx, student_logits in enumerate(exit_logits):
                    if student_idx < teacher_idx:
                        kd_loss += kd_loss_func(student_logits, teacher_logits) * (student_idx+1)
                        
                loss = (ce_loss + kd_loss)/(exit_num*(exit_num+1))
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

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
        