# import torch
# import torch.nn as nn
# import time

# class RandomLabelToHiddenStatus(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RandomLabelToHiddenStatus, self).__init__()
#         # 定义网络层
#         self.fc1 = nn.Linear(input_dim, 512)  # 第一个全连接层
#         self.fc2 = nn.Linear(512, 512)       # 第二个全连接层
#         self.fc3 = nn.Linear(512, output_dim) # 第三个全连接层，输出维度匹配Transformer的隐藏状态

#     def forward(self, x):
#         # 定义前向传播
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # 不应用激活函数，因为Transformer的隐藏状态可以是任何值
#         return x

# # 输入维度是100，输出维度是197*386
# input_dim = 100
# output_dim = 197 * 386

# # 创建模型实例
# model = RandomLabelToHiddenStatus(input_dim, output_dim)

# # 创建一个随机标签的输入张量，大小为32x100
# random_labels = torch.randint(0, 100, (32, input_dim)).float()
# random_labels = random_labels.to(1)

# # 通过模型获取隐藏状态
# model.to(1)
# time.sleep(20)
# hidden_states = model(random_labels)

# # 调整隐藏状态的形状以匹配Transformer的期望输入，即32x197x386
# hidden_states = hidden_states.view(32, 197, 386)

# print(hidden_states.shape)  # 输出应该是torch.Size([32, 197, 386])

# import torch

# # 假设我们有一个包含形状为 1 的张量的元组
# tensors = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
# weight = torch.tensor([2,3,4], dtype=torch.float)

# cat_tensor = torch.cat(tensors)
# s = torch.sum(weight*cat_tensor)

# print("Concatenated Tensor:\n", cat_tensor)
# print(s)

# import torch.optim as optim
# import torch

# # 假设我们有一个模型
# model = torch.nn.Linear(20, 50)

# # 创建第一个优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # 执行一些训练步骤...

# # 现在创建第二个优化器，可能与第一个优化器有不同的参数
# pseudo_optimizer = optim.SGD(model.parameters(), momentum=0.99)

# # 将第一个优化器的状态复制到第二个优化器
# pseudo_optimizer.load_state_dict(optimizer.state_dict())

# # 此时，pseudo_optimizer 将具有与 optimizer 相同的学习率和其他设置
# print("Learning rate:", pseudo_optimizer.param_groups[0]['lr'])

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 定义一个简单的模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 创建模型实例
# model = SimpleModel()

# # 创建数据和目标
# x = torch.randn(1, 10, requires_grad=True)
# y = torch.tensor([1], dtype=torch.long)

# # 前向传播
# output = model(x)

# # 计算损失
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output, y)

# # 第一次反向传播，保留计算图
# loss.backward(retain_graph=True)

# # 打印第一次梯度
# print("First-order gradients:")
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")

# # 假设我们需要计算Hessian矩阵的近似，进行第二次反向传播
# # 这里只是一个示例，实际上计算Hessian需要更复杂的步骤
# hessian_vector_product = torch.autograd.grad(
#     outputs=output,
#     inputs=x,
#     grad_outputs=torch.ones_like(output),
#     create_graph=True,
#     retain_graph=True
# )[0]

# # 打印Hessian向量乘积的结果
# print("\nHessian-vector product:")
# print(hessian_vector_product)

# # 清理不再需要的计算图
# del hessian_vector_product

# # 进行第二次反向传播，如果需要
# loss.backward()

# # 打印第二次梯度
# print("Second-order gradients:")
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")

# class GradientRescaleFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input)
#         ctx.gd_scale_weight = weight
#         output = input
#         return output
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors
#         grad_input = grad_weight = None
#         if ctx.needs_input_grad[0]:
#             grad_input = ctx.gd_scale_weight * grad_output
#         return grad_input, grad_weight


# gradient_rescale = GradientRescaleFunction.apply

# x = torch.tensor([1,2,3], dtype=torch.float, requires_grad=True)
# z = x*x
# # z = gradient_rescale(z, 0.5)
# l = torch.sum(z)
# l.backward()
# print(x.grad)

import torch
import time

# # 一个包含PyTorch张量的元组
# tensors = (torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4))

# # 求和
# total = sum(tensors)  # 直接使用sum，因为PyTorch重载了sum函数

# print(total)  # 输出: tensor(10)

# class A(nn.Module):
#     def __init__(self,):
#         super(A, self).__init__()
#         self.l1 = nn.Linear(5,2)
#         self.c1 = nn.Linear(2,1)
#         self.l2 = nn.Linear(2,2)
#         self.c2 = nn.Linear(2,1)

#     def forward(self, x):
#         z = self.l1(x)
#         c1 = self.c1(z)
#         z = self.l2(z)
#         c2 = self.c2(z)
#         return c1, c2
    
# a = A()
# t = torch.tensor([1, 2,3,4,5], dtype=torch.float32)
# y = a(t)
# y.backward()
# for n, p in a.named_parameters():
#     print(n, p.grad)
# q = torch.tensor([4,5,6,7,8], dtype=torch.float32)
# y = a(q)
# y.backward()
# for n, p in a.named_parameters():
#     print(n, p.grad)


# hidden_states = torch.rand(12,13,14)
# a = torch.rand(12,13,14)
# print(hidden_states)
# a[:, 0] = hidden_states[:, 0]
# print(a)

# from trainer.generator.generator import Generator_CIFAR, Generator_LATENT

# g = Generator_LATENT()
# g.train()
# g.to(2)
# op = torch.optim.Adam(g.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

# for _ in range(200):
#     y = torch.randint(0, 100, (32,)).to(2)
#     gen_latent, eps = g(y, )
#     loss = g.diversity_loss(eps, gen_latent)
#     loss.backward()
#     print(loss.item())
#     op.step()

# a = (torch.tensor([10, 20]), )
# print(sum(a[:-1]))

# x = torch.tensor([1,2,3,4,5], dtype=torch.float32)
# a = A()
# a.train()
# c1, c2 = a(x)

# c1.backward(retain_graph=True)
# for n, p in a.named_parameters():
#     print(n, p.grad)
# c2.backward()
# # print(x.grad)
# for n, p in a.named_parameters():
#     print(n, p.grad)
# import copy
# device = 'cpu'
# # mean = torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=torch.float32).to(device)
# # std = torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=torch.float32).to(device)
# images = torch.randint(0, 256, (2, 3, 32, 32)).float()
# images = images.view(-1, 3, 32, 32) / 255
# clone = copy.deepcopy(images)
# images_reshaped = images.view(-1, 3, 32, 32) / 255
# a = torch.nn.functional.interpolate(images_reshaped, size=(224,224), mode='bilinear', align_corners=False)


# a = (a-mean.view(1,3,1,1))/std.view(1,3,1,1)

# print(a.shape)


# from torchvision import transforms
# import numpy as np

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.Normalize(
#             (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
#             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#         ),
#     ])

# transform2 = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((224, 224)),
#         transforms.Normalize(
#             (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
#             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#         ),
#     ])



# images_transformed = [transform(image) for image in images]
# images_transformed1 = np.stack(images_transformed, axis=0, dtype=np.float32)


# images = images.numpy()
# images = np.reshape(images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
# images_transformed = [transform2(image) for image in images]
# images_transformed2 = np.stack(images_transformed, axis=0, dtype=np.float32)


# # print(a.numpy()-images_transformed)
# # print(a.numpy())
# print(np.array_equal(images_transformed1, images_transformed2))

# for _ in range(10):
#     a = torch.rand(3750, 197, 384).to('cuda:0')
#     print(torch.std(a, [0,2]))

# torch.manual_seed(1024)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# print(torch.rand(1))
# print(torch.rand(1))
# from test_1 import t
# t()

# import torch.nn as nn
# import gc

# class A(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super(A, self).__init__()
#         self.l = nn.Linear(1000, 10000)

# a = A()
# a.to(torch.device('cuda:2'))
# print('gpu')
# time.sleep(10)
# a.cpu()
# print('cpu')
# del a
# torch.cuda.empty_cache()
# gc.collect()
# time.sleep(30)



# from transformers import TFBertForPreTraining

# # 模型的路径，包含 ckpt.data 文件
# model_path = "/data/qvlehao/FL-EE/models/google-bert/bert-12-uncased/bert_model.ckpt.data-00000-of-00001"  # 替换为你的模型路径

# # 加载 TensorFlow 模型
# tf_model = TFBertForPreTraining.from_pretrained(model_path)

# import torch

# from transformers import BertModel

# # 将 TensorFlow 模型转换为 PyTorch 模型
# model = BertModel.from_pretrained(tf_model)
# # 保存模型
# model.save_pretrained("models/google-bert/bert-12-uncasedpytorch_model.bin")

# coding=utf-8
 
"""Convert BERT checkpoint."""
 
 
# import argparse
# import logging
 
# import torch
 
# a = torch.tensor([1.0,2.0], dtype=torch.float32, requires_grad=True)
# b = torch.tensor([3.0,4.0], dtype=torch.float32, requires_grad=True)
# c = torch.sum(a+b)
# c.backward()
# print(b.grad)


# from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
 
 
# logging.basicConfig(level=logging.INFO)
 
 
# def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
#     # Initialise PyTorch model
#     config = BertConfig.from_json_file(bert_config_file)
#     print("Building PyTorch model from configuration: {}".format(str(config)))
#     model = BertForPreTraining(config)
 
#     # Load weights from tf checkpoint
#     load_tf_weights_in_bert(model, config, tf_checkpoint_path)
 
#     # Save pytorch-model
#     print("Save PyTorch model to {}".format(pytorch_dump_path))
#     torch.save(model.state_dict(), pytorch_dump_path)
 
 
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument(
#         "--tf_checkpoint_path", default='/data/qvlehao/FL-EE/models/google-bert/bert-12-uncased/bert_model.ckpt', type=str, help="Path to the TensorFlow checkpoint path."
#     )
#     parser.add_argument(
#         "--bert_config_file",
#         default='/data/qvlehao/FL-EE/models/google-bert/bert-12-uncased/bert_config.json',
#         type=str,
#         help="The config json file corresponding to the pre-trained BERT model. \n"
#         "This specifies the model architecture.",
#     )
#     parser.add_argument(
#         "--pytorch_dump_path", default='pytorch_model.bin', type=str,  help="Path to the output PyTorch model."
#     )
#     args = parser.parse_args()
    
#     convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)

# import pickle as pkl

# for i in range(120):
#     with open(f"dataset/cifar100-224-d03-0.1/train/{i}.pkl", 'rb') as f:
#         l = pkl.load(f, encoding='bytes')
#         l = len(l[b'data'])
#         print(l)

import torch

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

device = 1

def adapt_batch(data):
    batch = {}
    for key in data.keys():
        batch[key] = data[key].to(device)
    label = batch['labels'].view(-1)
    return batch, label

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.total_num=120

config_path = 'models/google-bert/bert-12-uncased'
pre_model = Model.from_pretrained(pretrained_model_name_or_path=config_path)
eq_config = copy.deepcopy(pre_model.config)
        

eq_config.num_hidden_layers = 12

eq_exit_config = ExitConfig(eq_config, num_labels=2, exits=(2,5,8,11), policy='base', alg='exclusivefl') 
model = ExitModel(eq_exit_config)
model.load_state_dict(pre_model.state_dict(), strict=False)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
        'models/google-bert/bert-12-uncased',
        padding_side="right",
        model_max_length=128,
        use_fast=False,
    )

train_dataset = get_glue_dataset(args=args, path=f'dataset/glue/mrpc/train/', eval_valids=True)
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=None)
print(len(train_dataset))

valid_dataset = get_glue_dataset(args=args, path=f'dataset/glue/mrpc/valid/', eval_valids=True)
loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, collate_fn=None)
print(len(valid_dataset))

test_dataset = get_glue_dataset(args=args, path=f'dataset/glue/mrpc/test.pkl')
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=None)
print(len(test_dataset))


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
# optim = AdamW(params=optimizer_grouped_parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
optim = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

loss_func = nn.CrossEntropyLoss()
for epoch in range(100):
    batch_loss = []
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    # optim = torch.optim.Adam(params=optimizer_grouped_parameters, lr=1e-4*(0.99)**epoch, betas=(0.9, 0.999), eps=1e-08)
    optim = torch.optim.SGD(params=model.parameters(), lr=3e-5, momentum=0.9, weight_decay=1e-4)
    for idx, data in enumerate(loader_train):
        optim.zero_grad()

        batch, label = adapt_batch(data)

        exit_logits = model(**batch)[-1]
        
        # ws = [1 for _ in range(4)]
                    
        # exits_loss = ()
        # for i, exit_logits in enumerate(exits_logits):
        #     exits_loss += (loss_func(exit_logits, label) * ws[i],)
        
        # ce_loss = sum(exits_loss)
        ce_loss = loss_func(exit_logits, label)
        ce_loss.backward()
        optim.step()
        batch_loss.append(ce_loss.detach().cpu().item())
    
    print(sum(batch_loss) / len(batch_loss))
    model.eval()
    correct = 0
    total = 0
    corrects = [0 for _ in range(4)]

    with torch.no_grad():
        for data in loader_test:
            batch, labels = adapt_batch(data)
            
            exits_logits = model(**batch)
            
            for i, exit_logits in enumerate(exits_logits):
                _, predicted = torch.max(exit_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                corrects[i] += (predicted == labels).sum().item()
    acc = 100.00 * correct / total
    
    print(acc)
    acc_exits = [100 * c / (total/4) for c in corrects]
    print(acc, acc_exits)
    
# import torch
# import torch.nn as nn

# def kd_loss_func(pred, teacher):
#     kld_loss = nn.KLDivLoss(reduction='batchmean')
#     log_softmax = nn.LogSoftmax(dim=-1)
#     softmax = nn.Softmax(dim=1)
#     T=3
#     _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
#     return _kld

# a = torch.rand(1000,100)
# b = torch.rand(1000,100)
# print(kd_loss_func(a,b))