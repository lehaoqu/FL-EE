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
                for i, teacher_logits in enumerate(exit_logits):
                    for j, student_logits in enumerate(exit_logits):
                        if i == j: continue
                        else: 
                            print(kd_loss.item())
                            kd_loss += kd_loss_func(student_logits, teacher_logits) / (len(exit_logits)-1)
                loss = ce_loss + kd_loss
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        