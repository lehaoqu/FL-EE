import torch

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    return parser.parse_args()

class Client(BaseClient):
    def run(self):
        self.train()
    
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
                
                loss = torch.zeros(1).to(self.device)
                exit_logits = self.model(**batch)
                for logits in exit_logits:
                    loss += self.loss_func(logits, label.view(-1))
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
        