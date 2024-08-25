import torch.cuda

import torch
from torch import nn
import torch.nn.functional as F
from typing import *

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)

from transformers.models.vit.configuration_vit import ViTConfig as Config
from transformers.models.vit import ViTPreTrainedModel
from transformers.models.vit.modeling_vit import *
from transformers.models.vit.modeling_vit import ViTForImageClassification as Model

from transformers.models.bert.modeling_bert import *
from utils.modelload.model import BaseModule


class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output
        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply


class ExitConfig(ViTConfig):
    
    model_type = "vit"

    def __init__(
        self,
        config, 
        exits=None,
        num_labels=None,
        base_model=None,
        classifier_archi=None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            qkv_bias=config.qkv_bias,
            encoder_stride=config.encoder_stride,
            **kwargs
        )
        self.exits = exits if exits is not None else tuple(range(config.num_hidden_layers))
        self.classifier_dropout = 0.0
        self.num_labels = num_labels
        self.base_model = base_model
        self.classifier_archi = classifier_archi


# class ViTEncoderRee(nn.Module):
#     def __init__(self, config: ViTExitConfig) -> None:
#         super().__init__()
#         self.config = config
#         self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
#         self.gradient_checkpointing = False
#         # TODO more accurate ree config
#         self.accumulator = Ree(
#             recurrent_steps=1,
#             heads=8,
#             depth=1,
#             base_model=config.base_model,
#             num_classes=config.num_labels,
#             adapter=None,
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ) -> EncoderOutputRee:
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
        
#         cls_tokens = []
#         all_ree_exit_outputs = ()

#         for i, layer_module in enumerate(self.layer):    
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_head_mask = head_mask[i] if head_mask is not None else None

#             # TODO hidden_states需要修改，添加最开始的cls token
#             # 取出第位于0位置的 cls_tokens (batch*1*hidden_size)
#             cls_token_batch = hidden_states[:, 0][:, None, :]
#             cls_tokens.append(cls_token_batch)
#             # exit_cls_tokens是在1维上的cat (batch*exit_idx*hidden_size)
#             exit_cls_tokens = torch.cat((cls_tokens), 1)
            
#             # print(f'exit_cls_tokens: {exit_cls_tokens.shape}')
#             mod_tokens = self.accumulator(exit_cls_tokens)
#             _outputs = self.accumulator.head(mod_tokens[:, 0] + exit_cls_tokens[:, -1])
#             # 记录每个exit的logits  exits_num * (batch*label_nums)
#             all_ree_exit_outputs += (_outputs, )
#             # transformer hidden_status更新
#             layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

#             hidden_states = layer_outputs[0]

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         return EncoderOutputRee(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             ree_exit_outputs=all_ree_exit_outputs,
#         )


# class ViTModelRee(ViTPreTrainedModel):
#     def __init__(self, config: ViTExitConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
#         super().__init__(config)
#         self.config = config

#         self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
#         self.encoder = ViTEncoderRee(config)

#         self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.pooler = ViTPooler(config) if add_pooling_layer else None

#         # Initialize weights and apply final processing
#         self.post_init()
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

#     def get_input_embeddings(self) -> ViTPatchEmbeddings:
#         return self.embeddings.patch_embeddings

#     def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#         class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         interpolate_pos_encoding: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPooling]:

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if pixel_values is None:
#             raise ValueError("You have to specify pixel_values")

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
#         expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
#         if pixel_values.dtype != expected_dtype:
#             pixel_values = pixel_values.to(expected_dtype)

#         embedding_output = self.embeddings(
#             pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
#         )

#         # TODO ree 
#         # get batch 
#         B = embedding_output.shape[0]
#         # cls_tokens是一个batch中的多个同等地位的cls token
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         # 在embedding前面添加cls
#         # print(f'cls_tokens: {cls_tokens.shape}')
#         # print(f'e_output: {embedding_output.shape}')
#         cls_embedding_output = torch.cat((cls_tokens, embedding_output), dim=1)
#         # print(f'cls_e_output: {cls_embedding_output.shape}')
        
#         encoder_outputs: EncoderOutputRee = self.encoder(
#             # 传递cls_embedding_output
#             cls_embedding_output,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]
#         sequence_output = self.layernorm(sequence_output)
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
#             return head_outputs + encoder_outputs[1:]

#         return encoder_outputs.ree_exit_outputs
        

class ViTExitLayer(nn.Module):

    def __init__(self, config: ExitConfig, index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = index
        self.exit = True if index in config.exits else False
        if self.exit:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        policy: Optional[str]='base'
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        
        # exit
        if self.exit is True:
            exit_idx = self.config.exits.index(self.layer_index)
            if policy == 'base':
                logits = self.classifier(self.layernorm(layer_output)[:, 0, :])
            elif policy == 'boosted':
                layer_output = gradient_rescale(layer_output, 1.0/(len(self.config.exits) - exit_idx))
                logits = self.classifier(self.layernorm(layer_output)[:, 0, :])
                layer_output = gradient_rescale(layer_output, len(self.config.exits) - exit_idx - 1)
                
            outputs = (layer_output, logits, outputs)   
        else:
            outputs = (layer_output, None, outputs)
        
        return outputs


class ViTExitEncoder(nn.Module):
    
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTExitLayer(config, index) for index in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        exit_idxs: Optional[Tuple[int]] = None,
        policy:Optional[str]='base'
    ) -> Union[tuple, BaseModelOutput, torch.Tensor]:
        exits_logits = ()
        
        if latent is None:
            for i, layer_module in enumerate(self.layer):
                layer_head_mask = head_mask[i] if head_mask is not None else None
                layer_outputs = layer_module(hidden_states, layer_head_mask, policy)
                hidden_states, exit_logits = layer_outputs[0], layer_outputs[1]
                if layer_module.exit:
                    exits_logits += (exit_logits,)
            return exits_logits
        
        else:
            # == only return end_exit's logits
            begin_exit = exit_idxs[0]
            end_exit = exit_idxs[1]
            begin_layer = self.config.exits[begin_exit]+1 if begin_exit is not None else 0
            end_layer = self.config.exits[end_exit]
            layers = self.layer[begin_layer:end_layer+1]
            hidden_states = latent
            for layer_module in layers:
                layer_outputs = layer_module(hidden_states, None, policy)
                hidden_states, exit_logits = layer_outputs[0], layer_outputs[1]
            
            return exit_logits


class ViTExitModel(ViTPreTrainedModel):
    
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTExitEncoder(config)

        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        latent: Optional[torch.Tensor] = None,
        exit_idxs: Optional[Tuple[int]] = None,
        policy:Optional[str]='base'
    ) -> Union[Tuple, BaseModelOutputWithPooling, torch.Tensor]:
        
        if latent is None:
        
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
            if pixel_values.dtype != expected_dtype:
                pixel_values = pixel_values.to(expected_dtype)
            embedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )
            encoder_outputs = self.encoder(
                embedding_output,
                head_mask=head_mask,
                policy=policy
            )
            return encoder_outputs
        else:
            encoder_output = self.encoder(
                latent,
                exit_idxs=exit_idxs,
                head_mask=None,
                policy=policy
            )
            return encoder_output

class ExitModel(ViTPreTrainedModel, BaseModule):
    
    def __init__(self, config: ExitConfig) -> None:
        ViTPreTrainedModel.__init__(self, config)
        BaseModule.__init__(self,)

        self.num_labels = config.num_labels
        # if config.classifier_archi == 'ree':
        #     self.vit = ViTModelRee(config, add_pooling_layer=False)
        # else:
        #     self.vit = ViTModel(config, add_pooling_layer=False)
        #     for i in config.exits:
        #         setattr(self, f"classifier_{i}", nn.Linear(config.hidden_size, config.num_labels))
        self.vit = ViTExitModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        latent: Optional[torch.Tensor] = None,
        exit_idxs: Optional[Tuple[int]] = None,
        policy:Optional[str]='base'
    ) -> Union[tuple, ImageClassifierOutput, torch.Tensor]:
        if latent is None:
            outputs = self.vit(
                pixel_values,
                head_mask=head_mask,
                interpolate_pos_encoding=interpolate_pos_encoding,
                policy=policy
            )
            return outputs
        else:
            output = self.vit(
                latent=latent,
                exit_idxs=exit_idxs,
                policy=policy
            )
            return output

        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output[:, 0, :])
        
        # if self.config.classifier_archi == 'ree':
        #     all_logits = outputs
        #     logits = outputs[self.config.exits[-1]]
        # else:
        #     hidden_states = BaseModelOutputWithPooling(outputs).hidden_states
        #     logits = getattr(self, f"classifier_{self.config.exits[-1]}")(self.layernorm(hidden_states[self.config.exits[-1]])[:, 0, :])
            
        #     all_logits = tuple()
        #     for i in range(len(self.config.exits)):
        #         exit = self.config.exits[i]
        #         classifier = getattr(self, f"classifier_{exit}")
        #         all_logits += (classifier(self.layernorm(hidden_states[exit + 1])[:, 0, :]), )

        