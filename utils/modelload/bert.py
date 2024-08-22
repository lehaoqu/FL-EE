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


class Classifier(nn.Module):

    def __init__(self, config, input_size, labels_num):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, classifier_dropout)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)

        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, hidden, mask):
        mask = (mask > 0). \
                unsqueeze(1). \
                repeat(1, mask.shape[-1], 1). \
                unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_atten(hidden, hidden, hidden, mask)

        hidden = hidden[:, 0]

        output_1 = torch.tanh(self.dropout(self.output_layer_1(hidden)))
        logits = self.output_layer_2(self.dropout(output_1))
        return logits


class ExitModel(BertPreTrainedModel, BaseModule):
    
    def __init__(self, config: ExitConfig):
        BertPreTrainedModel.__init__(self, config)
        BaseModule.__init__(self,)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        for i in config.exits:
            if i < self.config.num_hidden_layers:
                setattr(self.bert.encoder.layer[i], f"classifier", Classifier(config, config.hidden_size, self.num_labels))

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
                
        hidden_states = BaseModelOutputWithPoolingAndCrossAttentions(outputs).hidden_states

        all_logits = tuple()
        for i in self.config.exits:
            if i < self.config.num_hidden_layers:
                classifier = getattr(self.bert.encoder.layer[i], f"classifier")
                all_logits += (classifier(hidden_states[i + 1], attention_mask),)

        return all_logits