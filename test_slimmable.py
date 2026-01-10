import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTEmbeddings, ViTPreTrainedModel
from utils.modelload.modelloader import load_model
from utils.modelload.slimmable import set_width_ratio, convert_to_slimmable, custom_ops_dict, set_model_config
import numpy as np
import random

from thop import profile

seed = 1117
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
class Test:
    def __init__(self):
        seed = 1117
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.dummy = {'pixel_values': torch.randn(1, 3, 224, 224).to(0)}
        self.ratios = [1.0, 0.75, 0.5, 0.25]

    def adma_optim(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optim = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=0.005, betas=(0.9, 0.999), eps=1e-08)
        return optim

    def test_slimmable_conv2d(self):
        data = torch.randn(1, int(64*1.0), 56, 56).to(0)
        
        # forward test
        conv = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(0)
        origin_conv = copy.deepcopy(conv)
        slim_conv = convert_to_slimmable(conv, ratios=self.ratios).to(0)
        set_width_ratio(1.0, slim_conv)
        
        origin_out = origin_conv(data)
        slim_out = slim_conv(data)
        
        assert torch.allclose(origin_out, slim_out, atol=1e-20), f"slimmable conv2d output does not match original output at ratio 1.0"
        print("slimmable conv2d output matches original output at ratio 1.0")

        # backward test
        optim_origin = torch.optim.SGD(origin_conv.parameters(), lr=0.01)
        optim_slim = torch.optim.SGD(slim_conv.parameters(), lr=0.01)
        for epoch in range(10):
            optim_origin.zero_grad()
            optim_slim.zero_grad()

            origin_out = origin_conv(data)
            slim_out = slim_conv(data)

            origin_loss = origin_out.sum()
            origin_loss.backward()
            
            slim_loss = slim_out.sum()
            slim_loss.backward()

            print('loss:', origin_loss.item(), slim_loss.item())
            assert torch.allclose(origin_loss, slim_loss, atol=1e-20), f"slimmable conv2d loss does not match original loss at ratio 1.0"
            print("slimmable conv2d loss matches original loss at ratio 1.0 in epoch", epoch)
            optim_origin.step()
            optim_slim.step()


    def test_slimmable_linear(self):
        data = torch.randn(1, int(512*1.0)).to(0)
        linear = nn.Linear(512, 1024).to(0)
        origin_linear = copy.deepcopy(linear)
        origin_out = origin_linear(data)
        slim_linear = convert_to_slimmable(linear, ratios=self.ratios).to(0)
        # print(linear)
        
        set_width_ratio(1.0, slim_linear)
        slim_out = slim_linear(data)
        assert torch.allclose(origin_out, slim_out, atol=1e-20), f"slimmable linear output does not match original output at ratio 1.0"
        print("slimmable linear output matches original output at ratio 1.0")

        # backward test
        optim_origin = torch.optim.SGD(origin_linear.parameters(), lr=0.01)
        optim_slim = torch.optim.SGD(slim_linear.parameters(), lr=0.01)
        for epoch in range(10):
            optim_origin.zero_grad()
            optim_slim.zero_grad()

            origin_out = origin_linear(data)
            slim_out = slim_linear(data)

            origin_loss = origin_out.sum()
            origin_loss.backward()
            
            slim_loss = slim_out.sum()
            slim_loss.backward()

            print('loss:', origin_loss.item(), slim_loss.item())
            assert torch.allclose(origin_loss, slim_loss, atol=1e-20), f"slimmable linear loss does not match original loss at ratio 1.0"
            print("slimmable linear loss matches original loss at ratio 1.0 in epoch", epoch)
            optim_origin.step()
            optim_slim.step()

    def test_slimmable_layernorm(self):
        data = torch.randn(1, int(512*1.0)).to(0)
        layernorm = nn.LayerNorm(512).to(0)
        origin_layernorm = copy.deepcopy(layernorm)
        origin_out = origin_layernorm(data)
        slim_layernorm = convert_to_slimmable(layernorm, ratios=self.ratios).to(0)
        # print(layernorm)
        
        set_width_ratio(1.0, slim_layernorm)
        slim_out = slim_layernorm(data)
        assert torch.allclose(origin_out, slim_out, atol=1e-20), f"slimmable layernorm output does not match original output at ratio 1.0"
        print("slimmable layernorm output matches original output at ratio 1.0")

        # backward test
        optim_origin = torch.optim.SGD(origin_layernorm.parameters(), lr=0.01)
        optim_slim = torch.optim.SGD(slim_layernorm.parameters(), lr=0.01)
        for epoch in range(10):
            optim_origin.zero_grad()
            optim_slim.zero_grad()

            origin_out = origin_layernorm(data)
            slim_out = slim_layernorm(data)

            origin_loss = origin_out.sum()
            origin_loss.backward()
            
            slim_loss = slim_out.sum()
            slim_loss.backward()

            print('loss:', origin_loss.item(), slim_loss.item())
            assert torch.allclose(origin_loss, slim_loss, atol=1e-20), f"slimmable layernorm loss does not match original loss at ratio 1.0"
            print("slimmable layernorm loss matches original loss at ratio 1.0 in epoch", epoch)
            optim_origin.step()
            optim_slim.step()


    def test_slimmable_vit(self):
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.config_path = './models/facebook/deit-tiny-patch16-224'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        args.slimmable = False

        ratios = [1.0, 0.75, 0.5, 0.25]
        depth = 12

        dummy = {'pixel_values': torch.randn(1, 3, 224, 224).to(0)}

        model = load_model(args, model_depth=depth, is_scalefl=False, exits=(2,5,8,11)).to(0)
        origin_model = copy.deepcopy(model)
        # print(origin_model)
        original_exits_logits = model(**dummy)
        # 转换为slimmable
        slim_model = convert_to_slimmable(model, ratios=ratios).to(0)
        # for name, para in origin_model.named_parameters():
        #     print(name, para.shape)
        # exit(0)
        # print(model)
        # 记录原始的 hidden size 和 intermediate size
        set_model_config(slim_model.config)

        set_width_ratio(1.0, slim_model)
        slim_exits_logits = slim_model(**dummy)

        for slim_logit, original_logit in zip(slim_exits_logits, original_exits_logits):
            # print(f"==={slim_logit.shape}==")
            # print(slim_logit)
            # print(original_logit)
            # print(torch.allclose(slim_logit, original_logit, atol=1e-20))
            assert torch.allclose(slim_logit, original_logit, atol=1e-20), f"slimmable vit output does not match original output at ratio 1.0"
        print("slimmable vit output matches original output at ratio 1.0")

        # backward test
        optim_origin = self.adma_optim(origin_model)
        optim_slim = self.adma_optim(slim_model)
        for epoch in range(10):
            optim_origin.zero_grad()
            optim_slim.zero_grad()

            origin_out = origin_model(**dummy)
            slim_out = slim_model(**dummy)

            origin_loss = sum([out.sum() for out in origin_out]).sum()
            origin_loss.backward()
            
            slim_loss = sum([out.sum() for out in slim_out]).sum()
            slim_loss.backward()

            print('loss:', origin_loss.item(), slim_loss.item())
            assert torch.allclose(origin_loss, slim_loss, atol=1e-6), f"slimmable vit loss does not match original loss at ratio 1.0"
            print("slimmable vit loss matches original loss at ratio 1.0 in epoch", epoch)
            optim_origin.step()
            optim_slim.step()




        # flops = {}
        # for ratio in ratios:
        #     # 设置宽度比例
        #     set_width_ratio(ratio, model)
        #     for depth in [3, 2, 1, 0]:
        #         from utils.train_utils import get_flops
        #         flops[(depth, ratio)] = get_flops(model, stop_exit=depth)
        # print(flops)

t = Test()
# t.test_slimmable_conv2d()
# t.test_slimmable_linear()
# t.test_slimmable_layernorm()
t.test_slimmable_vit()