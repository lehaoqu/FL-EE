import torch
from torch.utils.data import Dataset

import logging
from dataset.utils.dataset_utils import load_tsv, load_np, load_pkl

logger = logging.getLogger(__name__)

import transformers
transformers.logging.set_verbosity_error()


class GLUEClassificationDataset(Dataset):
    def __init__(self, args=None, path=None, tokenizer=None, need_process=False, eval_valids=False):
        self.path = path
        
        if eval_valids:
            dict_all = [load_pkl(f'{path}{i}.pkl') for i in range(args.total_num)]
            total_data = {}
            for key in dict_all[0].keys():
                for dic in dict_all:
                    total_data.setdefault(key, []).extend(dic[key])
            self.ann = total_data
        else: 
            self.ann = load_tsv(path) if need_process else load_pkl(path)
            # print(self.ann.keys())

            if need_process:
                labels = [item["label"] for item in self.ann]
                input_ids = [item["input_id"] for item in self.ann]
                self.ann = self.preprocess(labels, input_ids, tokenizer)
            else:
                self.ann['input_ids'] = [torch.tensor(row, dtype=torch.int) for row in self.ann['input_ids']]
                self.ann['attention_mask'] = [torch.tensor(row, dtype=torch.int) for row in self.ann['attention_mask']]
                self.ann['labels'] = [torch.tensor(row, dtype=torch.long) for row in self.ann['labels']]
        
        self.input_ids = self.ann['input_ids']
        self.labels = self.ann['labels']
        self.attention_mask = self.ann['attention_mask']

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = tokenizer(
                strings,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=128
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
        return len(self.labels)

    def __getitem__(self, index):
        return dict(
            input_ids=self.input_ids[index],
            labels=self.labels[index],
            attention_mask=self.attention_mask[index]
        )