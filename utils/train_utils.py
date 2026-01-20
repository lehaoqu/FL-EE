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


def get_layer_idx(name, ft='full'):
    layer_idx = 0
    if 'vit.encoder.layer' in name or 'bert.encoder.layer' in name:
        if ft == 'full':
            layer_idx = name.split('.')[3]
        elif ft == 'lora':
            layer_idx = name.split('.')[5]
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


def difficulty_measure(exits_logits, label=None, metric='loss', rt_exits_diff=False):
    if metric == 'loss':
        exits_loss = ()
        loss_func = nn.CrossEntropyLoss()
        for i, logits in enumerate(exits_logits):
            exits_loss += (loss_func(logits, label).unsqueeze(0),)
        diff_pred = min(sum(exits_loss)/len(exits_loss), torch.tensor([9.99]).to(exits_logits[0].device)) # TODO cifar glue
        exits_diff = torch.cat(exits_loss)
        
    elif metric == 'confidence':
        confidences = ()
        for logits in exits_logits:
            probs = F.softmax(logits, dim=0)
            confidence = probs.max(dim=0, keepdim=False)[0]
            confidences += (1-confidence.unsqueeze(0), )
        diff_pred = (sum(confidences)/len(exits_logits))
        exits_diff = torch.cat(confidences)
        
    elif metric == 'cosine':
        last_logits = exits_logits[-1].unsqueeze(0)
        diff_pred = 0
        for logits in exits_logits:
            exit_logits = logits.unsqueeze(0)
            diff_pred += nn.functional.cosine_similarity(exit_logits, last_logits, dim=1)
        diff_pred = (1-diff_pred/len(exits_logits))
        # TODO exits_diff
    
    if rt_exits_diff: return (diff_pred, exits_diff)
    else: return diff_pred


def get_flops(args, model, stop_exit=3):
    from utils.modelload.slimmable import convert_to_slimmable, custom_ops_dict, set_width_ratio
    from thop import profile
    # 定义一个专用的 Module 包装器（放在类内或外均可，这里建议放外面或作为内部类）
    class _ExitWrapper(nn.Module):
        def __init__(self, model, dummy_input, stop_exit):
            super().__init__()
            self.model = model
            self.dummy_input = dummy_input
            self.stop_exit = stop_exit

        def forward(self, _=None):
            return self.model(**self.dummy_input, stop_exit=self.stop_exit)

    if 'cifar' in args.dataset or 'imagenet' in args.dataset:
        dummy_input = {'pixel_values': torch.zeros(1, 3, 224, 224).to(args.device)}
    elif 'glue' in args.dataset or 'bert' in args.dataset:
        dummy_input = {
            'input_ids': torch.zeros(1, 128, dtype=torch.long).to(args.device),
            'attention_mask': torch.ones(1, 128, dtype=torch.long).to(args.device)
        }
    else:
        dummy_input = {'pixel_values': torch.zeros(1, 3, 224, 224).to(args.device)}

    # ✅ 使用 _ExitWrapper 包装（是 nn.Module！）
    wrapped_model = _ExitWrapper(model, dummy_input, stop_exit=stop_exit).to(args.device)

    # thop 需要一个 dummy input tensor（即使不用）
    dummy_tensor_for_thop = torch.zeros(1, 1).to(args.device)

    # ✅ 现在传入的是 nn.Module，不会报错
    macs, _ = profile(wrapped_model, inputs=(dummy_tensor_for_thop,), verbose=False, custom_ops=custom_ops_dict)
    return float(macs * 2)


def kd_loss_func(pred, teacher, T=1.0):
    kld_loss = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=1)
    _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
    return _kld


def area_under_fitted_curve(y_list, x_list, *, fit: str = 'spline', s: float = 0.0, k: int = 3,
                            bounds=None) -> float:
    """计算拟合曲线下的面积（AUC）。

    参数:
        y_list: 纵坐标列表（长度与 x_list 相同）。
        x_list: 横坐标列表（长度与 y_list 相同）。
        fit: 拟合方式，默认 'spline'（三次样条）；也可传 'trapz' 直接对原始点做梯形积分。
        s: UnivariateSpline 的平滑系数（s=0 表示插值拟合；s>0 会更平滑）。
        k: 样条阶数（默认 3；需满足 1 <= k <= 5 且点数 > k）。
        bounds: 积分区间 (xmin, xmax)。为 None 时使用 (min(x), max(x)).

    返回:
        拟合曲线在指定区间内的面积（float）。
    """
    import numpy as np

    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"x_list 与 y_list 长度必须一致，得到 {len(x)} vs {len(y)}")
    if x.size < 2:
        raise ValueError("至少需要 2 个点来计算面积")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x_list / y_list 里包含 NaN 或 Inf")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 去重：UnivariateSpline 需要严格递增的 x
    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        y_accum = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_accum, inverse, y)
        np.add.at(counts, inverse, 1.0)
        x = unique_x
        y = y_accum / np.maximum(counts, 1.0)

    if bounds is None:
        a = float(x[0])
        b = float(x[-1])
    else:
        a = float(bounds[0])
        b = float(bounds[1])
        if a > b:
            a, b = b, a

    if fit == 'trapz':
        mask = (x >= a) & (x <= b)
        xx = x[mask]
        yy = y[mask]
        if xx.size < 2:
            raise ValueError("bounds 区间内点数不足，无法进行积分")
        return float(np.trapz(yy, xx)), float(np.trapz(yy, xx)/(b - a))

    if fit != 'spline':
        raise ValueError("fit 仅支持 'spline' 或 'trapz'")

    # spline 拟合 + 解析积分（失败则回退到 trapz）
    try:
        from scipy.interpolate import UnivariateSpline

        kk = int(k)
        if not (1 <= kk <= 5):
            raise ValueError("k 需满足 1 <= k <= 5")
        if x.size <= kk:
            kk = max(1, min(kk, int(x.size - 1)))

        spl = UnivariateSpline(x, y, k=kk, s=float(s))
        return float(spl.integral(a, b)), float(spl.integral(a, b)/(b - a))
    except Exception:
        grid = np.linspace(a, b, num=max(200, int(x.size * 10)))
        yy = np.interp(grid, x, y)
        return float(np.trapz(yy, grid)), float(np.trapz(yy, grid)/(b - a))


def fuse_curves_take_max(slim_x, slim_y, *, ref_ratio: float = 1.0, grid: str = 'ref', num=None):
    """将多条曲线融合为一条“最大值包络”曲线。

    约定:
        - slim_x / slim_y 为 dict: ratio -> x_list/y_list
        - bounds 使用 ratio=1.0（默认 ref_ratio）那条曲线的 x 范围

    融合方式:
        - 以 ref_ratio 曲线的 x 作为默认网格（grid='ref'）
        - 其它曲线通过线性插值对齐到同一 x 网格
        - y 取各曲线的逐点最大值

    参数:
        slim_x: dict[ratio, list[float]]
        slim_y: dict[ratio, list[float]]
        ref_ratio: 用于确定 bounds/默认网格的 ratio（通常为 1.0）
        grid: 'ref' 使用 ref 曲线的 x 点；'linspace' 使用均匀网格
        num: 当 grid='linspace' 时，网格点数；为 None 时使用 len(ref_x)

    返回:
        (x_fused, y_fused): 两个 list[float]
    """
    import numpy as np

    if not isinstance(slim_x, dict) or not isinstance(slim_y, dict):
        raise TypeError("slim_x / slim_y 需要是 dict: ratio -> list")

    def _pick_key(dct, target_ratio: float):
        if target_ratio in dct:
            return target_ratio
        if str(target_ratio) in dct:
            return str(target_ratio)
        if int(target_ratio) in dct:
            return int(target_ratio)
        candidates = []
        for k in dct.keys():
            try:
                candidates.append((abs(float(k) - float(target_ratio)), k))
            except Exception:
                continue
        if not candidates:
            raise KeyError(f"未找到 ref_ratio={target_ratio} 对应的曲线")
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _prepare_xy(x_list, y_list):
        x_arr = np.asarray(x_list, dtype=float)
        y_arr = np.asarray(y_list, dtype=float)
        if x_arr.shape != y_arr.shape:
            raise ValueError("某条曲线的 x/y 长度不一致")
        if x_arr.size < 2:
            return None
        if not (np.isfinite(x_arr).all() and np.isfinite(y_arr).all()):
            return None

        order = np.argsort(x_arr)
        x_arr = x_arr[order]
        y_arr = y_arr[order]

        unique_x, inverse = np.unique(x_arr, return_inverse=True)
        if unique_x.size != x_arr.size:
            y_accum = np.zeros_like(unique_x, dtype=float)
            counts = np.zeros_like(unique_x, dtype=float)
            np.add.at(y_accum, inverse, y_arr)
            np.add.at(counts, inverse, 1.0)
            x_arr = unique_x
            y_arr = y_accum / np.maximum(counts, 1.0)
        return x_arr, y_arr

    ref_key = _pick_key(slim_x, ref_ratio)
    if ref_key not in slim_y:
        raise KeyError(f"ref_ratio={ref_ratio} 在 slim_y 中不存在")

    ref_xy = _prepare_xy(slim_x[ref_key], slim_y[ref_key])
    if ref_xy is None:
        raise ValueError("ref_ratio 曲线点数不足或包含非法值")
    ref_x, ref_y = ref_xy

    a = float(ref_x[0])
    b = float(ref_x[-1])
    if a > b:
        a, b = b, a

    if grid == 'ref':
        x_grid = ref_x
    elif grid == 'linspace':
        n = int(num) if num is not None else int(ref_x.size)
        n = max(2, n)
        x_grid = np.linspace(a, b, num=n)
    else:
        raise ValueError("grid 仅支持 'ref' 或 'linspace'")

    ys = []
    for ratio_key in slim_x.keys():
        if ratio_key not in slim_y:
            continue
        xy = _prepare_xy(slim_x[ratio_key], slim_y[ratio_key])
        if xy is None:
            continue
        x_i, y_i = xy
        y_interp = np.interp(x_grid, x_i, y_i, left=np.nan, right=np.nan)
        ys.append(y_interp)

    if not ys:
        raise ValueError("没有可用的曲线用于融合")

    y_fused = np.nanmax(np.stack(ys, axis=0), axis=0)
    return x_grid.astype(float).tolist(), y_fused.astype(float).tolist()
