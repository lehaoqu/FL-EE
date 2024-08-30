import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def add_args(parser):
    parser.add_argument('--meta_gap', type=int, default=5, help="meta gap")
    parser.add_argument('--meta_lr', type=float, default=1e-4, help="meta lr")
    parser.add_argument('--meta_weight_decay', type=float, default=1e-4, help="meta weight_decay")
    parser.add_argument('--meta_p', type=int, default=15, help="meta valid p")
    return parser

class MetaSGD(torch.optim.SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for name, parameter in self.net.named_parameters():
            grad = grads[name]
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))



class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)


    def forward(self, x):
        return F.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=1, output_size=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class MLP_tanh(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=1, output_size=1):
        super(MLP_tanh, self).__init__()
        self.first_hidden_layer = HiddenLayer(input_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.tanh(x)


class Policy():
    def __init__(self, args) -> None:
        self.name = 'l2w'
        self.exits_num = args.exits_num
        self.device = args.device
        # == default input is loss or confidence   [exits_num] ==
        self.meta_net = MLP_tanh(input_size=self.exits_num, output_size=self.exits_num).to(self.device)
        
        self.loss_func = nn.CrossEntropyLoss()
        self.meta_optimizer = torch.optim.Adam(self.meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
        self.target_probs = self.calc_target_probs()[args.meta_p-1]
        

    def calc_target_probs(self,):
        for p in range(1, 40):
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.arange(1, self.exits_num+1))
            probs /= probs.sum()
            if p == 1:
                probs_list = probs.unsqueeze(0)
            else:
                probs_list = torch.cat((probs_list, probs.unsqueeze(0)), 0)
        
        return probs_list
    
    
    def train(self, model, batch, label, ws=None) -> torch.tensor:
        exits_logits = model(**batch)
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'

        ws = [1 for i in range(self.exits_num)] if ws is None else ws
        
        for i, exit_logits in enumerate(exits_logits):
            loss_vector = F.cross_entropy(exit_logits, label, reduction='none')
            if i==0:
                losses = loss_vector.unsqueeze(1)
            else:
                losses = torch.cat((losses, loss_vector.unsqueeze(1)), dim=1)
                
        with torch.no_grad():
            weight = self.meta_net(losses)
            weight = weight - torch.mean(weight, 1, keepdim=True) 
            # TODO 0.8
            weight = torch.ones(weight.shape).to(weight.device) + 0.8 * weight
        
        # print(f'weight for exits: {weight[0]}')
        losses_tensor = torch.mean(weight*losses, 0)
        losses_tuple = tuple(losses_tensor)
        exits_loss = ()
        for i, exit_loss in enumerate(losses_tuple) :
            exits_loss += (exit_loss * ws[i],)
        return exits_loss, exits_logits
    
    
    def train_meta(self, model, batch, label, optimizer):
        batch_1, batch_2 = {},{}
        for key in batch.keys():
            batch_1[key], batch_2[key] = batch[key].chunk(2, dim=0)
        label_1, label_2 = label.chunk(2, dim=0)
        
        data = [batch_1, batch_2, label_1, label_2]
        self.train_meta_part(model, optimizer, data)
        
        data = [batch_2, batch_1, label_2, label_1]
        self.train_meta_part(model, optimizer, data)
        
    
    def train_meta_part(self, model, optimizer, data):
        # TODO p = 15
        
        
        pseudo_net = copy.deepcopy(model).to(self.device)
        pseudo_net.train()
        batch_pseudo, batch_meta, label_pseudo, label_meta = data
        # print(batch_pseudo)
        exits_logits = pseudo_net(**batch_pseudo)
        # print(exits_logits)
        for i, exit_logits in enumerate(exits_logits):
            pseudo_loss_vector = F.cross_entropy(exit_logits, label_pseudo, reduction='none')
            if i==0:
                pseudo_losses = pseudo_loss_vector.unsqueeze(1)
            else:
                pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
  
        pseudo_weight = self.meta_net(pseudo_losses.detach())
        pseudo_weight = pseudo_weight - torch.mean(pseudo_weight, 1, keepdim=True) 
        # TODO 0.8
        pseudo_weight = torch.ones(pseudo_weight.shape).to(pseudo_weight.device) + 0.8 * pseudo_weight
        pseudo_loss_multi_exits = torch.sum(torch.mean(pseudo_weight * pseudo_losses, 0))
        # print(pseudo_loss_multi_exits)
        pseudo_loss_multi_exits.backward(retain_graph=True)
        # pseudo_grads = {n: p.grad for n, p in pseudo_net.named_parameters()}
        
        # pseudo_grads = torch.autograd.grad(pseudo_loss_multi_exits, pseudo_net.parameters(), create_graph=True, allow_unused=True)

        # for n, p in pseudo_net.named_parameters():
        #     if p.grad is None:
        #         print(n)
        
        # pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters())
        # pseudo_optimizer.load_state_dict(optimizer.state_dict())
        # pseudo_optimizer.meta_step(pseudo_grads)
        pseudo_optimizer = copy.deepcopy(optimizer)
        pseudo_optimizer.step()
        
        # del pseudo_grads
        
        for n, p in pseudo_net.named_parameters():
            if p.requires_grad is False:
                print(n)
        meta_outputs = pseudo_net(**batch_meta)
        # print(meta_outputs)
        used_index = []
        meta_loss = 0.0
        
        total = 0
        correct = 0
        
        for j in range(self.exits_num):
            with torch.no_grad():
                confidence_target = F.softmax(meta_outputs[j], dim=1)  
                max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)  
                _, sorted_idx = max_preds_target.sort(dim=0, descending=True)  
                n_target = sorted_idx.shape[0]
                
                if j == 0:
                    selected_index = sorted_idx[: math.floor(n_target * self.target_probs[j])]
                    selected_index = selected_index.tolist()
                    used_index.extend(selected_index)
                elif j < self.exits_num - 1:
                    filter_set = set(used_index)
                    unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                    selected_index = unused_index[: math.floor(n_target * self.target_probs[j])]  
                    used_index.extend(selected_index)
                else:
                    filter_set = set(used_index)
                    selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
            if len(selected_index) > 0:
                
                exit_logits = meta_outputs[j][selected_index]
                labels = label_meta[selected_index].long()
                _, predicted = torch.max(exit_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                meta_loss += F.cross_entropy(meta_outputs[j][selected_index], label_meta[selected_index].long(), reduction='mean')

        # print(f'{100*correct/(total):.2f}, {correct}, {total}')
        
        self.meta_optimizer.zero_grad()
        # print(meta_loss)
        meta_loss.backward()
        self.meta_optimizer.step()
        # print(meta_loss)
    

    def __call__(self, exits_logits):
        return exits_logits
    
    # == for finetune in server == 
    def sf(self, exits_logits):
        return exits_logits[-1]