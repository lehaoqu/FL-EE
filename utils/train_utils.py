import math
import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F

class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


class HardDarkRank(nn.Module):
    def __init__(self, alpha=2, beta=3, permute_len=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


def get_layer_idx(name):
    layer_idx = 0
    if 'vit.encoder.layer' in name or 'bert.encoder.layer' in name:
        layer_idx = name.split('.')[3]
    return int(layer_idx)


def crop_tensor_dimensions(tensor, origin_target):
    """
    裁剪张量中指定大小的维度到新的尺寸。
    
    参数:
    - tensor: 要裁剪的原始张量。
    - target_sizes: 一个包含需要裁剪的维度大小的列表。
    - new_size: 新的维度大小。
    
    返回:
    - cropped_tensor: 裁剪后的张量。
    """
    # 找到所有需要裁剪的维度的索引
    indices_to_crop = [i for i, size in enumerate(tensor.shape) if size in origin_target.keys()]
    
    # 裁剪每个找到的维度
    cropped_tensor = tensor
    for index in indices_to_crop:
        # 确保我们不会裁剪超出原始尺寸的范围
        crop_size = min(origin_target[tensor.shape[index]], tensor.shape[index])
        cropped_tensor = cropped_tensor.narrow(index, 0, crop_size)
    
    return cropped_tensor


def aggregate_scale_tensors(tensors, samples, device):
        
    def zero_pad(a, new_shape):
        expanded_a = torch.zeros(new_shape, dtype=a.dtype).to(device)
        start_indices = tuple(0 for _ in range(len(new_shape)))
        end_indices = a.shape
        index_tensor = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
        expanded_a[index_tensor] = a
        return expanded_a
            
    def get_size(tensor):
        size = 1
        for s in tensor.shape:
            size *= s
        return size
    
    weights = [torch.full(tensor.shape, sample).to(device) for (tensor, sample) in zip(tensors, samples)]
    sizes = [get_size(tensor) for tensor in tensors]
    max_shape = tensors[sizes.index(max(sizes))].shape
    
    global_tensor = torch.zeros(max_shape).to(device)
    global_weight = torch.zeros(max_shape).to(device)
    
    for idx, tensor in enumerate(tensors):
        weighted_tensor = tensor * weights[idx]
        weighted_tensor = zero_pad(weighted_tensor, max_shape)
        global_tensor += weighted_tensor
        
        weight = zero_pad(weights[idx], max_shape)
        global_weight += weight
    
    global_tensor = global_tensor / global_weight
    return global_tensor
        

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def calc_target_probs(exits_num):
    for p in range(1, 40):
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        probs = torch.exp(torch.log(_p) * torch.arange(1, exits_num+1))
        probs /= probs.sum()
        if p == 1:
            probs_list = probs.unsqueeze(0)
        else:
            probs_list = torch.cat((probs_list, probs.unsqueeze(0)), 0)
    
    return probs_list


def exit_policy(exits_num, exits_logits, target_probs):
    used_index, selected_index_list = [], []
    for j in range(exits_num):
        with torch.no_grad():
            confidence_target = F.softmax(exits_logits[j], dim=1)  
            max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)  
            _, sorted_idx = max_preds_target.sort(dim=0, descending=True)  
            n_target = sorted_idx.shape[0]
            
            if j == 0:
                selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                selected_index = selected_index.tolist()
                used_index.extend(selected_index)
            elif j < exits_num - 1:
                filter_set = set(used_index)
                unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                selected_index = unused_index[: math.floor(n_target * target_probs[j])]  
                used_index.extend(selected_index)
            else:
                filter_set = set(used_index)
                selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
        
        if len(selected_index) > 0:
            selected_index_list.append(selected_index)
    return selected_index_list


def difficulty_measure(exits_logits, label=None, metric='loss', exits_diff=False):
    if metric == 'loss':
        exits_loss = ()
        loss_func = nn.CrossEntropyLoss()
        for i, logits in enumerate(exits_logits):
            exits_loss += (loss_func(logits, label),)
        diff_pred = min(sum(exits_loss)/len(exits_loss) * 2, torch.tensor(9.99).to(label.device)) # TODO cifar glue
        exits_diff = torch.tensor([min((exit_loss*2).detach(), torch.tensor(9.99).to(label.device)) for exit_loss in exits_loss]).to(label.device)
        
    elif metric == 'confidence':
        confidences = ()
        for logits in exits_logits:
            probs = F.softmax(logits, dim=0)
            confidence = probs.max(dim=0, keepdim=False)[0]
            confidences += confidence
        diff_pred = (1-sum(confidences)/len(exits_logits))*5
        exits_diff = torch.tensor([exit_confidence.detach() for exit_confidence in confidences]).to(exit_logits[0].device)
        
    elif metric == 'cosine':
        last_logits = exits_logits[-1].unsqueeze(0)
        diff_pred = 0
        for logits in exits_logits:
            exit_logits = logits.unsqueeze(0)
            diff_pred += nn.functional.cosine_similarity(exit_logits, last_logits, dim=1)
        diff_pred = (1-diff_pred/len(exits_logits))*5
        # TODO exits_diff
    
    if exits_diff: return (diff_pred, exits_diff)
    else: return diff_pred


