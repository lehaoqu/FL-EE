
import torch
import importlib, json
from transformers import AutoTokenizer
from dataset.utils.dataset_utils import load_tsv, load_np, load_pkl
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.bert import *
import copy, argparse
from dataset import (
    get_cifar_dataset,
    get_glue_dataset
)
from utils.train_utils import AdamW
from utils.options import args_parser


def adapt_batch(data):
    batch = {}
    for key in data.keys():
        batch[key] = data[key].to(device)
    label = batch['labels'].view(-1)
    return batch, label

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# args.total_num = 120
# args.device = device
# 
args = args_parser()
args.exits_num = 4
device = args.device
if not os.path.exists(f'./{args.suffix}'):
    os.makedirs(f'./{args.suffix}')
    
model_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.pth'
config_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.json'    
output_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.txt' 
output = open(output_path, 'a')  

config_path = args.config_path
pre_model = Model.from_pretrained(pretrained_model_name_or_path=config_path)
eq_config = copy.deepcopy(pre_model.config)
        

eq_config.num_hidden_layers = 12

eq_exit_config = ExitConfig(eq_config, num_labels=2, exits=(2,5,8,11), policy='base', alg='exclusivefl') 
model = ExitModel(eq_exit_config)
model.load_state_dict(pre_model.state_dict(), strict=False)
model.to(device)

with open(config_save_path, 'w', encoding='utf-8') as f:
    json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

tokenizer = AutoTokenizer.from_pretrained(
        'models/google-bert/bert-12-uncased',
        padding_side="right",
        model_max_length=128,
        use_fast=False,
)

ds = args.dataset
train_dataset = get_glue_dataset(args=args, path=f'dataset/glue/{ds}/train/', eval_valids=True)
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
print(len(train_dataset))

valid_dataset = get_glue_dataset(args=args, path=f'dataset/glue/{ds}/valid/', eval_valids=True)
loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=False, collate_fn=None)
print(len(valid_dataset))

# test_dataset = get_glue_dataset(args=args, path=f'dataset/glue/{ds}/test.pkl')
# loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=None)
# print(len(test_dataset))


optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

loss_func = nn.CrossEntropyLoss()

policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
policy = policy_module.Policy(args)
best_acc = 0.0

for epoch in range(50):
    batch_loss = []
    model.train()
    # optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    for idx, data in enumerate(loader_train):
        
        optim.zero_grad()

        batch, label = adapt_batch(data)
        
        if policy.name == 'l2w' and idx % args.meta_gap == 0:
            policy.train_meta(model, batch, label, optim)

        exits_ce_loss, _ = policy.train(model, batch, label)
        ce_loss = sum(exits_ce_loss)
        ce_loss.backward()
        optim.step()
        batch_loss.append(ce_loss.detach().cpu().item())
        
    
    print(sum(batch_loss) / len(batch_loss))
    model.eval()
    correct = 0
    total = 0
    corrects = [0 for _ in range(4)]

    if epoch % 1 == 0:
        with torch.no_grad():
            for data in loader_valid:
                batch, labels = adapt_batch(data)
                
                exits_logits = model(**batch)
                exits_logits = policy(exits_logits)
                
                for i, exit_logits in enumerate(exits_logits):
                    _, predicted = torch.max(exit_logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    corrects[i] += (predicted == labels).sum().item()
        acc = 100.00 * correct / total
        
        if acc > best_acc:
            best_acc = acc
            model.save_model(model_save_path)
            
        
        acc_exits = [100 * c / (total/4) for c in corrects]
        output.write(f'{epoch}, {acc}, {acc_exits}\n')
        output.flush()
        print(epoch, acc, acc_exits)