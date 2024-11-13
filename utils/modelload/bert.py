import torch.cuda

import torch
from torch import nn
import torch.nn.functional as F
from typing import *

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.vit.modeling_vit import *
from transformers.models.bert.configuration_bert import BertConfig as Config
from transformers.models.bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import *
from transformers.models.bert.modeling_bert import BertForSequenceClassification as Model
from utils.modelload.model import BaseModule

from utils.modelload.model import Ree


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



# === Bert ===
class ExitConfig(BertConfig):

    model_type = "bert-exit"

    def __init__(
            self,
            config,
            exits=None,
            num_labels=None,
            base_model=None,
            classifier_archi=None,
            policy=None,
            alg=None,
            **kwargs,
    ):
        super().__init__(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
            position_embedding_type=config.position_embedding_type,
            use_cache=config.use_cache,
            classifier_dropout=config.classifier_dropout,
            **kwargs,
        )
        self.exits = exits if exits is not None else tuple(range(config.num_hidden_layers))
        self.base_model = base_model
        self.num_labels = num_labels if num_labels is not None else 2
        self.classifier_atchi = classifier_archi
        self.policy = policy
        self.alg = alg


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        # print(scores.shape)
        # print(mask.shape)
        # print(scores[0][0])
        scores = scores + mask
        # print(scores[0][0])
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)

        return output


# class Classifier(nn.Module):

#     def __init__(self, config, input_size, labels_num):
#         super(Classifier, self).__init__()
#         self.input_size = input_size
#         self.cla_hidden_size = 128
#         self.cla_heads_num = 2
#         self.labels_num = labels_num
#         self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, classifier_dropout)
#         self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
#         self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)

#         self.dropout = nn.Dropout(classifier_dropout)

#     def forward(self, hidden, mask):
#         mask = (mask > 0). \
#                 unsqueeze(1). \
#                 repeat(1, mask.shape[-1], 1). \
#                 unsqueeze(1)
#         mask = mask.float()
#         mask = (1.0 - mask) * -10000.0

#         hidden = torch.tanh(self.output_layer_0(hidden))
#         hidden = self.self_atten(hidden, hidden, hidden, mask)

#         hidden = hidden[:, 0]

#         output_1 = torch.tanh(self.dropout(self.output_layer_1(hidden)))
#         logits = self.output_layer_2(self.dropout(output_1))
#         return logits


class BertExitLayer(nn.Module):
    def __init__(self, config: ExitConfig, index: int):
        super().__init__()
        self.config = config
        self.layer_index = index
        self.exit = True if index in config.exits else False
        if self.exit:
            self.classifier_pooler = BertPooler(config)
            self.classifier_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        o_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        f = None
        if self.exit is True:
            exit_idx = self.config.exits.index(self.layer_index)
            if self.config.policy == 'base' or self.config.policy == 'l2w':
                f = self.classifier_dropout(self.classifier_pooler(layer_output))
                logits = self.classifier(f)
            elif self.config.policy == 'boosted':
                layer_output = gradient_rescale(layer_output, 1.0/(len(self.config.exits) - exit_idx))
                f = self.classifier_dropout(self.classifier_pooler(layer_output))
                logits = self.classifier(f)
                layer_output = gradient_rescale(layer_output, len(self.config.exits) - exit_idx - 1)
            
        
            outputs = (layer_output, logits, outputs, f)
        else:
            outputs = (layer_output, None, outputs, f)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertExitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertExitLayer(config, index) for index in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        o_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        stop_exit: Optional[int] = None
    ):
        
        exits_logits = ()
        exits_feature = ()

        for i, layer_module in enumerate(self.layer):

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                o_attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
            )

            hidden_states, exit_logits, f = layer_outputs[0], layer_outputs[1], layer_outputs[3]
            if layer_module.exit:
                exits_logits += (exit_logits, )
                exits_feature += (f, )
            if stop_exit is not None and i == self.config.exits[stop_exit]: break

        return exits_logits, exits_feature
    
    
class BertExitEncoderRee(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.accumulator = Ree(
            recurrent_steps=1,
            heads=8,
            modulation=True,
            exit_head='normlinear',
            mode='add',
            base_model='bt',
            num_classes=100,
            adapter=None,
            depth=1,
            attn_dim=16,
            mlp_ratio=1.35,

        )
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        o_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        stop_exit: Optional[int] = None
    ):
        
        cls_tokens = []
        exits_logits = ()
        exits_feature = ()

        for i, layer_module in enumerate(self.layer):

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
            )
            
            hidden_states = layer_outputs[0]
            # hidden_states = gradient_rescale(hidden_states, 1.0/(len(self.config.exits) - i))
            
            cls_token_batch = hidden_states[:, 0][:, None, :]
            cls_tokens.append(cls_token_batch)
            mod_tokens = None
            
            if i in self.config.exits:
                # TODO hidden_states需要修改，添加最开始的cls token
                # 取出第位于0位置的 cls_tokens (batch*1*hidden_size)

                # exit_cls_tokens是在1维上的cat (batch*exit_idx*hidden_size)
                exit_cls_tokens = torch.cat((cls_tokens), 1)
                
                # print(f'exit_cls_tokens: {exit_cls_tokens.shape}')
                mod_tokens = self.accumulator(exit_cls_tokens)
                _outputs = self.accumulator.head(mod_tokens[:, 0] + exit_cls_tokens[:, -1])
                # 记录每个exit的logits  exits_num * (batch*label_nums)
                exits_logits += (_outputs, )
                exits_feature += (hidden_states[:, 0], )
            if self.accumulator.modulation:
                if mod_tokens is None:
                    mod_tokens = self.accumulator(torch.cat((cls_tokens), 1))
                    hidden_states[:, 0] = mod_tokens[:, -1]
            if stop_exit is not None and i == self.config.exits[stop_exit]: break

            # hidden_states = gradient_rescale(hidden_states, len(self.config.exits) - i - 1)

        return exits_logits, exits_feature


class BertExitModel(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertExitEncoder(config) if self.config.alg != 'reefl' else BertExitEncoderRee(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_latent: Optional[bool] = False,
        stop_exit:Optional[int] = None,
        rt_embedding:Optional[bool] = False
    ):
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_ids.size())
        
        if is_latent is False:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # past_key_values_length
            past_key_values_length = 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            
        
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            hidden_states = embedding_output
        else:
            hidden_states = input_ids
        
        if rt_embedding:
            return hidden_states
        
        exits_logits, exits_feature = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            o_attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            stop_exit=stop_exit
        )
        return (exits_logits, exits_feature)
        # sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )


class ExitModel(BertPreTrainedModel, BaseModule):
    
    def __init__(self, config: ExitConfig):
        BertPreTrainedModel.__init__(self, config)
        BaseModule.__init__(self, config.exits)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertExitModel(config, add_pooling_layer=False)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )

        # self.dropout = nn.Dropout(classifier_dropout)

        # for i in config.exits:
        #     if i < self.config.num_hidden_layers:
        #         setattr(self.bert.encoder.layer[i], f"classifier", Classifier(config, config.hidden_size, self.num_labels))

        # self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_latent: Optional[bool] = False,
        stop_exit:Optional[int] = None,
        rt_embedding:Optional[bool]=False,
        labels:Optional[torch.Tensor] = None,
        rt_feature:Optional[bool]=False,
    ):
        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                stop_exit=stop_exit,
                is_latent=is_latent,
                rt_embedding=rt_embedding
            )
        if rt_embedding: return outputs
        if rt_feature: return outputs
        return outputs[0]
                
        # hidden_states = BaseModelOutputWithPoolingAndCrossAttentions(outputs).hidden_states

        # all_logits = tuple()
        # for i in self.config.exits:
        #     if i < self.config.num_hidden_layers:
        #         classifier = getattr(self.bert.encoder.layer[i], f"classifier")
        #         all_logits += (classifier(hidden_states[i + 1], attention_mask),)
