import copy
import importlib
from multiprocessing import dummy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTEmbeddings, ViTPreTrainedModel
from eval import Eval
from utils.modelload.modelloader import load_model
from utils.modelload.slimmable import set_width_ratio, convert_to_slimmable, custom_ops_dict, set_model_config
import numpy as np
import random
import json

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
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        args.device = 0
        self.args = args


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


    def test_slimmable_layerexit(self):


        from utils.train_utils import get_flops
        from utils.modelload.vit import ViTExitLayer, ExitConfig, ExitModel
        config_path = './models/facebook/deit-tiny-patch16-224'
        based_model = importlib.import_module(f'utils.modelload.vit')
        pre_model = based_model.Model.from_pretrained(pretrained_model_name_or_path=config_path)
        eq_config = copy.deepcopy(pre_model.config)
        num_labels = 100
        exits = (2,5,8,11)
        policy = 'base'
        alg = 'depthfl'
        blocks = (2,5,8,11)
        eq_exit_config = ExitConfig(eq_config, num_labels=num_labels, exits=exits, policy=policy, alg=alg, blocks=blocks) 

        origin_layer = ViTExitLayer(config=eq_exit_config, index=2).to(0)
        slim_layer = copy.deepcopy(origin_layer)
        convert_to_slimmable(slim_layer, ratios=self.ratios).to(0)

        # print(slim_layer)
        # exit(0)
        set_model_config(slim_layer.config)
        set_width_ratio(0.75, slim_layer)
        
        # slim_flops = get_flops(self.args, slim_layer, input_feature=True)
        origin_flops = get_flops(self.args, origin_layer, input_feature=True)
        
        for ratio in self.ratios:
            set_width_ratio(ratio, slim_layer)
            slim_flops = get_flops(self.args, slim_layer, input_feature=True)
            print(f"At ratio {ratio}, Slimmable layer FLOPs: {slim_flops}, div by origin FLOPs: {slim_flops/origin_flops:.4f}")
       

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



    def test_slimmable_vit_lora(self):
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.config_path = './models/facebook/deit-tiny-patch16-224'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'lora'
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
        # print(slim_model)
        # for n, p in slim_model.named_parameters():
        #     print(n, p.shape, p.requires_grad)
        # exit(0)

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

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            origin_out = origin_model(**dummy)
            # print(len(origin_out), origin_out[0].shape)
            # print(origin_model)
            # print('======')
            # print(slim_model)
            # print(dict(origin_model.named_parameters())['base_model.model.vit.encoder.layer.2.classifier.modules_to_save.default.weight'].requires_grad)
            # exit(0)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            slim_out = slim_model(**dummy)

            origin_loss = sum([out.sum() for out in origin_out]).sum()
            origin_loss.backward()
            
            slim_loss = sum([out.sum() for out in slim_out]).sum()
            slim_loss.backward()


            print('loss:', origin_loss.item(), slim_loss.item())
            assert torch.allclose(origin_loss, slim_loss, atol=1e-6), f"slimmable vit loss does not match original loss at ratio 1.0"
            print("slimmable vit loss matches original loss at ratio 1.0 in epoch", epoch)
            
            for n, p in slim_model.named_parameters():
                if n not in dict(origin_model.named_parameters()):
                    continue
                p_origin = dict(origin_model.named_parameters())[n]
                if not torch.allclose(p, p_origin, atol=1e-6):
                    print(f"Parameter {n} does not match between slim and origin model.")
                if p.requires_grad:
                    if 'original_module' in n: continue
                    if p_origin.requires_grad is False:
                        print(f"Gradient of parameter {n} is None in origin model but not in slim model.")
                        continue
                    grad_origin = p_origin.grad
                    if p.grad == None: print(f"slim {n} grad is None")
                    if p_origin.grad == None: print(f"origin {n} grad is None")
                    if not torch.allclose(p.grad, grad_origin, atol=1e-6):
                        print(f"Gradient of parameter {n} does not match between slim and origin model.")
                        print(p.grad)
                        print(grad_origin)


            optim_origin.step()
            optim_slim.step()

            # for n, p in slim_model.named_parameters():
            #     if n not in dict(origin_model.named_parameters()):
            #         continue
            #     p_origin = dict(origin_model.named_parameters())[n]
            #     if not torch.allclose(p, p_origin, atol=1e-6):
            #         print(f"After step, Parameter {n} does not match between slim and origin model.")
            
            # exit(0)


    def test_slimmable_vit_reefl(self):
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.config_path = './models/facebook/deit-tiny-patch16-224'
        args.policy = 'boosted'
        args.alg = 'reefl'
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
        
        # print(slim_model)
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


    def test_slimmable_load(self):
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.policy = 'boosted'
        args.alg = 'reefl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        
        from utils.modelload.modelloader import load_model_eval
        config_path = 'EXPS2/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.25].json'
        model_path = 'EXPS2/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.25].pth'
        model = load_model_eval(args, model_path=model_path, config_path=config_path)
        # set_width_ratio(0.25, model)
        print(model)


    def test_slimmable_load_lora(self):
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.policy = 'boosted'
        args.alg = 'eefl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'lora'
        args.config_path = '/home/qvlehao/FL-EE/models/facebook/deit-tiny-patch16-224'
        
        from utils.modelload.modelloader import load_model_eval
        config_path = 'EXPS2/BASE_CIFAR_ALL_DY/lora_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0].json'
        model_path = 'EXPS2/BASE_CIFAR_ALL_DY/lora_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0].pth'

        model = load_model_eval(args, model_path=model_path, config_path=config_path)
        # set_width_ratio(0.25, model)
        print(model)


    def test_slimmbale_flops(self):
        from utils.train_utils import get_flops
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        args.device = 0
        
        from utils.modelload.modelloader import load_model_eval
        config_path = 'EXPS2/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.25].json'
        model_path = 'EXPS2/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.25].pth'
        slim_model = load_model_eval(args, model_path=model_path, config_path=config_path)
        
        config_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
        model_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
        origin_model = load_model_eval(args, model_path=model_path, config_path=config_path)
        
        set_width_ratio(1.0, slim_model)
        
        for depth in [0,1,2,3]:
            slim_flops = get_flops(args, slim_model, stop_exit=depth)
            origin_flops = get_flops(args,origin_model, stop_exit=depth)

            assert slim_flops == origin_flops, f"slimmable vit flops {slim_flops} does not match original flops {origin_flops} at ratio 1.0 and depth {depth}"
            print(f"slimmable vit flops {slim_flops} matches original flops {origin_flops} at ratio 1.0 and depth {depth}")



        # flops = {}
        # for ratio in ratios:
        #     # 设置宽度比例
        #     set_width_ratio(ratio, model)
        #     for depth in [3, 2, 1, 0]:
        #         from utils.train_utils import get_flops
        #         flops[(depth, ratio)] = get_flops(model, stop_exit=depth)
        # print(flops)


    def test_area(self):
        origin_eval_path = '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_eval.json'
        dct = json.loads(open(origin_eval_path, 'r').read())
        x = dct['flops']
        y = dct['test']
        from utils.train_utils import area_under_fitted_curve
        area, acc = area_under_fitted_curve(y, x)
        print(f"area: {area}, acc: {acc}")


    def test_slim_area(self):
        ratio_paths = {
            1.0: '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR_R/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.9]_slim_1.0_eval.json',
            0.95: '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR_R/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.9]_slim_0.9_eval.json',
        }

        # ratio_paths = {
        #     1.0: '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR_R/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.95-0.9]_slim_1.0_eval.json',
        #     0.95: '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR_R/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.95-0.9]_slim_0.95_eval.json',
        #     0.9: '/home/qvlehao/FL-EE/front-exps/BASE_CIFAR_R/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.95-0.9]_slim_0.9_eval.json',
        # }
        slim_x_list = {}
        slim_y_list = {}
        for ratio, path in ratio_paths.items():
            dct = json.loads(open(path, 'r').read())
            slim_x_list[ratio] = dct['flops']
            slim_y_list[ratio] = dct['test']

        from utils.train_utils import fuse_curves_take_max, area_under_fitted_curve
        fx, fy = fuse_curves_take_max(slim_x_list, slim_y_list, ref_ratio=1.0)
        area, acc = area_under_fitted_curve(fy, fx)
        print(f"slimmable area: {area}, acc: {acc}")

    def test_slim_dynamic_compute(self):
        '''
        Docstring for test_slim_dynamic_compute
        比较slim训练的模型和原始浅层模型的效果。
        如：0.9的slim和只有3个出口的原始模型
        :param self: Description
        '''
        class A: pass
        args = A()
        args.model = 'vit'
        args.dataset = 'cifar100_noniid1000'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        args.device = 0
        from utils.modelload.modelloader import load_model_eval
        slim_config_path = 'EXPS2/BASE_CIFAR_ALL/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.9].json'
        slim_model_path = 'EXPS2/BASE_CIFAR_ALL/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[1.0-0.9].pth'
        origin_config_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
        origin_model_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'

        slim_model = load_model_eval(args, model_path=slim_model_path, config_path=slim_config_path)        
        origin_model = load_model_eval(args, model_path=origin_model_path, config_path=origin_config_path, model_depth=9)

        set_width_ratio(0.9, slim_model)

        eval = Eval(args=args)
        eval.eval(slim_model_path, slim_config_path, model=slim_model)

        
        eval = Eval(args=args)
        eval.eval(origin_model_path, origin_config_path, model=origin_model)





t = Test()
# t.test_slimmable_conv2d()
# t.test_slimmable_linear()
# t.test_slimmable_layernorm()
# t.test_slimmable_vit()
# t.test_slimmable_vit_reefl()

# t.test_slimmable_load()
# t.test_slimmbale_flops()

# t.test_area()
# t.test_slim_area()

# t.test_slimmable_layerexit()
# t.test_slimmable_vit()
t.test_slimmable_vit_lora()
# t.test_slimmable_load_lora()
# t.test_slim_dynamic_compute()
