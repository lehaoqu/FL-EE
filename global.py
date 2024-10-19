
import torch
import importlib, json
from utils.modelload.bert import *
from dataset import (
    get_cifar_dataset,
    get_glue_dataset,
    get_svhn_dataset
)
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset
from utils.options import args_parser
from utils.modelload.modelloader import load_model
import numpy as np
import random
from tqdm import tqdm


def adapt_batch(data, args):
    batch = {}
    for key in data.keys():
        batch[key] = data[key].to(device)
        if key == 'pixel_values':
            if 'cifar' in args.dataset:
                batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
            else:
                batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
    label = batch['labels'].view(-1)
    return batch, label

args = args_parser()
args.exits_num = 4
device = args.device

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if not os.path.exists(f'./{args.suffix}'):
    os.makedirs(f'./{args.suffix}')
    
model_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.pth'
config_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.json'    
output_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.txt' 
output = open(output_path, 'a')  


model = load_model(args, model_depth=12, is_scalefl=False, exits=(2,5,8,11))
model.to(device)

with open(config_save_path, 'w', encoding='utf-8') as f:
    json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=4)

ds = args.dataset
if 'cifar' in ds:
    get_dataset = get_cifar_dataset
elif 'svhn' in ds:
    get_dataset = get_svhn_dataset
else:
    ds = f'glue/{ds}'
    get_dataset = get_glue_dataset
    
train_dataset = get_dataset(args=args, path=f'dataset/{ds}/train/', eval_valids=True)
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
print(len(train_dataset))

valid_dataset = get_dataset(args=args, path=f'dataset/{ds}/valid/', eval_valids=True)
loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
print(len(valid_dataset))

# test_dataset = get_glue_dataset(args=args, path=f'dataset/glue/{ds}/test.pkl')
# loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=None)
# print(len(test_dataset))


optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [150, 225], gamma=0.1)
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'gamma', 'beta']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#     'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
# ]
# optim = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08)


loss_func = nn.CrossEntropyLoss()

policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
policy = policy_module.Policy(args)

best_acc = 0.0
for epoch in range(50):
    batch_loss = []
    model.train()
    
    for idx, data in enumerate(loader_train):
        
        optim.zero_grad()

        batch, label = adapt_batch(data, args)
        
        if policy.name == 'l2w' and idx % args.meta_gap == 0:
            policy.train_meta(model, batch, label, optim)

        exits_ce_loss, _ = policy.train(model, batch, label)
        ce_loss = sum(exits_ce_loss)
        ce_loss.backward()
        optim.step()
        batch_loss.append(ce_loss.detach().cpu().item())
        
    print(sum(batch_loss) / len(batch_loss))
    
    # scheduler.step()
    
    model.eval()
    correct = 0
    total = 0
    corrects = [0 for _ in range(4)]

    if epoch % 1 == 0:
        with torch.no_grad():
            for data in loader_valid:
                batch, labels = adapt_batch(data, args)
                
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