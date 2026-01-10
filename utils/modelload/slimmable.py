from typing import Optional, Tuple, Union
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTEmbeddings, ViTConfig

# 全局变量控制当前宽度比例，默认为 1.0 (大模型)
CURRENT_WIDTH_RATIO = 1.0
HIDEEN_SIZE = 192  # 根据具体模型调整
INTERMEDIATE_SIZE = 768  # 根据具体模型调整
NUM_ATTENTION_HEADS = 3  # 根据具体模型调整


def set_model_config(config):
    global HIDEEN_SIZE, INTERMEDIATE_SIZE, NUM_ATTENTION_HEADS
    HIDEEN_SIZE = config.hidden_size
    INTERMEDIATE_SIZE = config.intermediate_size
    NUM_ATTENTION_HEADS = config.num_attention_heads

def set_width_ratio(ratio, model):
    def get_aligned_dim(origin_all_head_size, ratio):
        return int(origin_all_head_size * ratio // NUM_ATTENTION_HEADS * NUM_ATTENTION_HEADS)

    global CURRENT_WIDTH_RATIO
    CURRENT_WIDTH_RATIO = ratio

    for module in model.modules():
        if isinstance(module, ViTSelfAttention):
            # 1. 计算当前的 hidden_dim (必须对齐 head_dim)
            # 使用我们之前定义的 get_aligned_dim
            current_all_head_size = get_aligned_dim(module.original_all_head_size, ratio)
            
            # 2. 计算对应的head size
            current_attention_head_size = current_all_head_size // module.num_attention_heads
            
            # 3. 【直接修改属性】覆盖 HuggingFace 对象中的值
            module.all_head_size = current_all_head_size
            module.attention_head_size = current_attention_head_size
            
            # 验证：确保数学逻辑成立，防止报错
            if module.all_head_size != module.num_attention_heads * module.attention_head_size:
                raise ValueError("切片后的维度无法被 head_size 整除，请检查对齐逻辑！")
    


class SlimmableLinear(nn.Linear):
    def forward(self, input):
        # 获取当前权重的最大维度
        in_features = self.in_features
        out_features = self.out_features
        
        # 计算当前应该使用的维度
        # 注意：这里假设 input 的最后一维就是当前的 in_dim
        # in_dim = input.shape[-1]
        # out_dim = int(out_features * CURRENT_WIDTH_RATIO)

        if in_features == HIDEEN_SIZE or in_features == INTERMEDIATE_SIZE:
            in_dim = int(in_features * CURRENT_WIDTH_RATIO // NUM_ATTENTION_HEADS * NUM_ATTENTION_HEADS)
        else:
            in_dim = in_features
        if out_features == HIDEEN_SIZE or out_features == INTERMEDIATE_SIZE:
            out_dim = int(out_features * CURRENT_WIDTH_RATIO // NUM_ATTENTION_HEADS * NUM_ATTENTION_HEADS)
        else:
            out_dim = out_features
        
        # print(f'SlimmableLinear: in_dim={in_dim}, out_dim={out_dim}')
        # 权重切片：取前 out_dim 行，前 in_dim 列
        weight = self.weight[:out_dim, :in_dim]
        
        # 偏置切片
        bias = self.bias[:out_dim] if self.bias is not None else None
        
        return F.linear(input, weight, bias)


class SwitchableLayerNorm(nn.Module):
    def __init__(self, original_dim, eps=1e-12, ratios=[1.0, 0.5]):
        super().__init__()
        self.ratios = ratios
        self.eps = eps
        
        # 使用 ModuleDict 存储不同宽度对应的独立 LayerNorm
        # 键必须是字符串，所以我们将 ratio 转为 string
        self.norm_dict = nn.ModuleDict()
        for r in ratios:
            dim = int(original_dim * r)
            self.norm_dict[str(r).replace('.', 'p')] = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # CURRENT_WIDTH_RATIO 是我们在全局设置的变量
        ratio = CURRENT_WIDTH_RATIO 
        
        # 找到对应的 LN 层
        # 注意：浮点数转字符串可能存在精度问题，最好在设置 ratio 时统一格式
        # 这里简单处理，假设传入的 ratio 严格匹配 keys
        key = str(ratio).replace('.', 'p') 
        
        if key not in self.norm_dict:
            # 如果是任意动态宽度（非预设），则不得不退化回切片模式（但在训练固定宽度时推荐用独立的）
            # 或者抛出错误
            raise ValueError(f"Ratio {ratio} not in initialized ratios {self.ratios}")
            
        return self.norm_dict[key](x)


class SlimmableConv2d(nn.Conv2d):
    def forward(self, input):
        # ViT 的 Patch Embeddings 使用 Conv2d
        # input shape: [B, C_in, H, W]
        # output channels (hidden_size) 需要切片
        in_channels = input.shape[1]
        out_channels = int(self.out_channels * CURRENT_WIDTH_RATIO)
        
        weight = self.weight[:out_channels, :in_channels, :, :]
        bias = self.bias[:out_channels] if self.bias is not None else None
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)



import torch
import torch.nn as nn
from thop import profile

# 引入你的自定义类
# from your_file import SlimmableLinear, SwitchableLayerNorm 
def count_slimmable_linear(m, x, y):
    # x 是输入 tuple (input,)
    # y 是输出 tensor
    # 矩阵乘法的 MACs = output_elements * input_dimension
    # y.numel() 包含了 batch_size * seq_len * out_dim
    # x[0].shape[-1] 是当前的 in_dim (切片后的)
    
    total_mul = y.numel() * x[0].shape[-1]
    
    # 加上 bias 的加法 (通常 FLOPs 只看乘加，或者乘法。thop 默认 Linear 只算乘法)
    # total_ops = total_mul + y.numel() 
    
    m.total_ops += torch.DoubleTensor([int(total_mul)])


def count_switchable_layernorm(m, x, y):
    # LayerNorm 的 FLOPs 很少，通常忽略，或者近似为 input_elements
    # 标准公式大约是 x.numel() * 2 (减均值除方差) + x.numel() * 2 (scale & shift)
    # 这里简单按元素个数计算
    m.total_ops += torch.DoubleTensor([int(x[0].numel())])


# 如果你也用了 SlimmableConv2d
def count_slimmable_conv2d(m, x, y):
    # x: input, y: output
    # Kernel size logic
    kernel_ops = m.weight.shape[2:].numel() # kernel_h * kernel_w
    # in_channels (切片后的)
    in_channels = x[0].shape[1]
    
    # MACs = Output Elements * (Kernel_size * In_Channels)
    total_ops = y.numel() * (in_channels * kernel_ops)
    m.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops_dict = {
    SlimmableLinear: count_slimmable_linear,
    # 如果用了 SwitchableLayerNorm
    SwitchableLayerNorm: count_switchable_layernorm,
    SlimmableConv2d: count_slimmable_conv2d,
}


def convert_to_slimmable(model, ratios=[1.0, 0.5]):
    """
    递归将 HuggingFace ViT 模型转换为 Slimmable 版本
    """

    for module in model.modules():
        if isinstance(module, ViTSelfAttention):
            # 备份原始参数，防止多次调整 ratio 后丢失基准
            if not hasattr(module, 'origin_attention_head_size'):
                module.origin_attention_head_size = module.attention_head_size
            if not hasattr(module, 'original_all_head_size'):
                module.original_all_head_size = module.all_head_size

    # 1. 替换基础层
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 替换 Linear
            new_layer = SlimmableLinear(module.in_features, module.out_features, bias=module.bias is not None)
            new_layer.weight.data = module.weight.data
            if module.bias is not None:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)
        
        elif isinstance(module, nn.LayerNorm):
            # 获取原有的维度
            original_dim = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
            
            # 初始化 Switchable LayerNorm
            new_layer = SwitchableLayerNorm(original_dim, eps=module.eps, ratios=ratios)
            
            # 将大模型(1.0)的权重复制给对应的层，小模型(0.5)的权重随机初始化
            new_layer.norm_dict['1p0'].weight.data = module.weight.data
            new_layer.norm_dict['1p0'].bias.data = module.bias.data
            
            # (可选) 如果你想用切片初始化小模型的LN作为起点也可以，但之后它们会独立更新
            # 小模型的初始化策略很重要，通常直接随机或复制切片均可
            for r in ratios:
                if r != 1.0:
                    dim = int(original_dim * r)
                    new_layer.norm_dict[str(r).replace('.', 'p')].weight.data = module.weight.data[:dim].clone()
                    new_layer.norm_dict[str(r).replace('.', 'p')].bias.data = module.bias.data[:dim].clone()

            setattr(model, name, new_layer)
            
        elif isinstance(module, nn.Conv2d):
            # 替换 Patch Embeddings 里的 Conv2d
            new_layer = SlimmableConv2d(module.in_channels, module.out_channels, 
                                        kernel_size=module.kernel_size, stride=module.stride, 
                                        padding=module.padding)
            new_layer.weight.data = module.weight.data
            if module.bias is not None:
                new_layer.bias.data = module.bias.data
            setattr(model, name, new_layer)
            
        else:
            # 递归处理子模块
            convert_to_slimmable(module, ratios)

    # 2. 特殊处理 Position Embeddings (如果它是 Parameter 而不是 Layer)
    # HF ViT 通常有一个 self.embeddings.position_embeddings
    # 我们需要在 forward 钩子中处理切片，或者修改 Embeddings 类的 forward
    # 简单起见，我们对 ViTEmbeddings 进行 Monkey Patch
    
    def slimmable_embeddings_forward(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=False, **kwargs):
        """
        修改后的 Embeddings forward 函数，增加了 **kwargs 以兼容不同版本的 transformers
        """
        # 1. 正常计算 patch embeddings
        # 注意：某些版本的 patch_embeddings 可能也接受 bool_masked_pos，
        # 但标准的 nn.Conv2d (我们替换后的 SlimmableConv2d) 只接受 input。
        # 这里我们只传 pixel_values 即可。
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 2. 处理 bool_masked_pos (用于 MAE 等掩码预训练任务)
        # 如果你的任务是分类，通常这里是 None。如果有值，我们需要根据 mask 调整 embeddings
        if bool_masked_pos is not None:
            seq_len = embeddings.shape[1]
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            embeddings = embeddings * (1 - w) + mask_token * w

        # 3. 添加位置编码 (核心修改点：切片)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # [关键修改] 对 position_embeddings 进行切片以匹配当前的 hidden_size
            # self.position_embeddings shape: [1, num_patches + 1, hidden_size]
            
            # 获取当前 embeddings 的实际宽度 (可能是 768, 或者是 384 等)
            current_dim = embeddings.shape[-1]
            
            # 对 position_embeddings 的最后一维进行切片
            # 注意：这里假设 position_embeddings 已经包含了 cls_token 的位置
            # 如果长度不匹配（比如输入图片尺寸变了），可能还需要对第二维切片，但通常分类任务图片尺寸固定
            pos_embed = self.position_embeddings[:, :embeddings.shape[1], :current_dim]
            
            embeddings = embeddings + pos_embed

        # 4. Dropout 和 LayerNorm (如果有)
        embeddings = self.dropout(embeddings)
        return embeddings

    def slimmable_transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (x.size()[-1]//self.attention_head_size, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 应用 Monkey Patch
    for module in model.modules():
        if isinstance(module, ViTSelfAttention):
            # 替换 Attention 的 reshape 逻辑
            module.transpose_for_scores = slimmable_transpose_for_scores.__get__(module, ViTSelfAttention)
            # module.forward = slimmable_vit_self_attention_forward.__get__(module, ViTSelfAttention)
        # if isinstance(module, ViTEmbeddings):
        #     # 替换 Embeddings 的 forward
        #     module.forward = slimmable_embeddings_forward.__get__(module, ViTEmbeddings)
    return model