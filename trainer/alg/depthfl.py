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
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                loss = torch.tensor(0.0).to(self.device)
                for logits in self.model(image):
                    loss += self.loss_func(logits, label)
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
        