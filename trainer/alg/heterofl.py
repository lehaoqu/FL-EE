import torch

from trainer.baseHFL import BaseServer, BaseClient

def add_args(parser):
    return parser.parse_args()

class Client(BaseClient):
    def run(self):
        self.train()
    
    def train(self):
        # === train ===
        self.model.to(self.device)
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, (image, label) in enumerate(self.loader_train):
                self.optim.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                last_logits = self.model(image)[-1]
                loss = self.loss_func(last_logits, label)
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))
    
    def clone_model(self, target):
        p_tensors = target.parameters_to_tensor(is_split=True)
        idx = self.args.eq_depths.index(self.eq_depth)
        self.model.tensor_to_parameters(torch.cat(p_tensors[:idx+1], 0))


class Server(BaseServer):
    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        avg_eq_tensor = [sum(eq_tensors) for eq_tensors in self.received_params]
        avg_tensor = torch.cat(avg_eq_tensor, 0)
        self.global_model.tensor_to_parameters(avg_tensor)