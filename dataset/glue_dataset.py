import copy
import json

import torch
from torch.utils.data import Dataset
import pandas as pd
from configs.training import train_config
import logging
from dataset.utils.dataset_utils import load_tsv

logger = logging.getLogger(__name__)

import transformers
transformers.logging.set_verbosity_error()


class GLUEClassificationDataset(Dataset):
    def __init__(self, path, tokenizer, partition="all"):
        list_dic = load_tsv(path)
        self.ann = list_dic
        if partition == "all":
            self.ann = self.ann
        elif partition == "train":
            self.ann = self.ann[:2*len(self.ann)//3]
        elif partition == "val":
            self.ann = self.ann[2*len(self.ann)//3:]
        elif partition == "test":
            pass

        labels = [item["label"] for item in self.ann]
        input_ids = [item["input_id"] for item in self.ann]

        data_dict = self.preprocess(labels, input_ids, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = tokenizer(
                strings,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=train_config.max_length
            )
        input_ids = labels = [
            tokenized_list.input_ids[i] for i in range(len(strings))
        ]
        input_ids_lens = labels_lens = [
            tokenized_list.input_ids[i].ne(tokenizer.pad_token_id).sum().item()
            for i in range(len(strings))
        ]
        attention_mask = [
            tokenized_list.attention_mask[i] for i in range(len(strings))
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
            attention_mask=attention_mask
        )

    def preprocess(self, labels, input_ids, tokenizer):

        input_ids_tokenized = self._tokenize_fn(input_ids, tokenizer)

        input_ids = input_ids_tokenized["input_ids"]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        attention_mask = input_ids_tokenized["attention_mask"]

        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        return dict(
            input_ids=self.input_ids[index],
            labels=self.labels[index],
            attention_mask=self.attention_mask[index]
        )