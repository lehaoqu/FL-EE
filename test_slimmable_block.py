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
from utils.modelload.modelloader import load_model_eval
from utils.dataloader_utils import load_dataset_loader

from thop import profile
from tqdm import tqdm

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
        args.dataset = 'cifar100_noniid1000'
        args.policy = 'boosted'
        args.alg = 'depthfl'
        args.blocks = (2,5,8,11)
        args.load_path = ''
        args.ft = 'full'
        args.device = 0
        args.bs = 32
        args.epoch = 20 # 设置训练轮次
        args.sr = 0.1
        args.total_num = 100
        args.valid_ratio = 0.2
        args.img_dir = './imgs'
        args.suffix = 'test'
        args.config_path = './models/facebook/deit-tiny-patch16-224'
        args.ensemble_weight = 0.2
        self.args = args
        args.slimmable = False

        # Load datasets as in baseHFL
        # Loading all clients' data and concatenating them
        self.total_clients = args.total_num
        all_train_datasets = []
        for i in range(self.total_clients):
            dataset, _ = load_dataset_loader(args=args, file_name='train', id=i)
            all_train_datasets.append(dataset)
        
        # 使用 ConcatDataset 合并所有 client 的数据
        from torch.utils.data import ConcatDataset, DataLoader
        self.dataset_train = ConcatDataset(all_train_datasets)
        self.loader_train = DataLoader(self.dataset_train, batch_size=args.bs, shuffle=True)
        
        self.dataset_valid, self.loader_valid = load_dataset_loader(args=args, eval_valids=True, shuffle=False)
        self.dataset_test, self.loader_test = load_dataset_loader(args=args, file_name='test', shuffle=False)
        
        args.exits_num = 4
        policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
        policy = policy_module.Policy(args)
        

        ratios = [1.0, 0.75, 0.5, 0.25]
        depth = 12

        config_path = '/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
        model_path = '/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR_ORIGIN/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
        model = load_model_eval(args, model_path=model_path, config_path=config_path)
        
        origin_model = copy.deepcopy(model)
        origin_model.eval()
        origin_model.to(0)

        # 转换为slimmable
        model = load_model(args, model_depth=depth, is_scalefl=False, exits=(2,5,8,11))
        slim_model = convert_to_slimmable(model, ratios=ratios).to(0)
        set_model_config(slim_model.config)
        
        # 补全 test_slimmable_vit: 参考 darkflpa2 的 train 函数
        # 1. 设置模型状态
        slim_model.train()
        origin_model.eval()
        for n, p in slim_model.named_parameters():
            print(n, p.shape, p.requires_grad)
        
        # 2. 模拟训练过程 (Block-wise KD)
        slim_model.train()
        param_optimizer = list(slim_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=0.05, betas=(0.9, 0.999), eps=1e-08)
        from utils.train_utils import kd_loss_func
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.epoch):
            # 每 5 个 epoch 进行一次评估
            # if (epoch) % 5 == 0:
            

            epoch_loss = 0
            # 遍历合并后的训练集
            for idx, data in tqdm(enumerate(self.loader_train), total=len(self.loader_train)):
                if (idx+1) % 500 == 0:
                    self.eval(model=slim_model)
                
                from dataset.cifar100_dataset import CIFARClassificationDataset
                batch = {'pixel_values': data['pixel_values'].to(0)}
                if 'cifar' in self.args.dataset:
                    batch['pixel_values'] = CIFARClassificationDataset.transform_for_vit(batch['pixel_values'])
                label = data['labels'].view(-1).to(0)
                
                # --- Teacher (origin_model) Pass ---
                with torch.no_grad():
                    outputs = origin_model(**batch, rt_full_feature=True) 
                    exits_full_logits = outputs[0] if isinstance(outputs, (list, tuple)) else []
                    exits_full_features = outputs[2] if isinstance(outputs, (list, tuple)) and len(outputs) > 1 else []
                    full_embeddings = origin_model(**batch, rt_embedding=True)

                optimizer.zero_grad()
                
                # --- Slim Model Passes ---
                set_width_ratio(1.0, slim_model)
                slim_outputs = policy.train(slim_model, batch, label, rt_full_feature=True)
                
                ce_loss = sum(slim_outputs[0])  # 交叉熵损失
                    
                # --- Slim Model KD Pass using origin_model targets ---
                kd_loss = torch.zeros(1).to(0)
                exits_num = len(slim_model.config.exits)
                
                for block_index in range(exits_num):
                    for slim_ratio in ratios:
                        if slim_ratio == 1.0:
                            continue
                        set_width_ratio(slim_ratio, slim_model)
                        
                        input_feature = full_embeddings.detach() if block_index == 0 else exits_full_features[block_index-1].detach()
                        batch['pixel_values'] = input_feature
                        
                        block_outputs = slim_model(**batch, is_latent=True, input_block=block_index, stop_exit=block_index)
                        
                        block_slim_logit = block_outputs[0][0]
                        block_slim_feature = block_outputs[2][0] 
                        
                        # print(block_slim_feature.shape, exits_full_features[block_index].shape)
                        feature_kd_loss = nn.MSELoss()(block_slim_feature, exits_full_features[block_index].detach())
                        logit_kd_loss = kd_loss_func(block_slim_logit, exits_full_logits[block_index].detach(), T=3.0)
                        
                        kd_loss += (feature_kd_loss + logit_kd_loss) / (len(ratios) - 1)
                
                # total_loss = ce_loss + kd_loss
                total_loss = ce_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            print(f"Epoch {epoch+1}/{self.args.epoch} - Loss: {epoch_loss/len(self.loader_train):.4f}")
            

        
        print(f"Test Training Done.")
        
    def eval(self, model=None):
        # 参考 baseHFL 添加评估函数
        print("Starting Evaluation (patterns from baseHFL)...")
        self.args.device = 0
        
        # 使用 self.loader_test 数据
        test_loader = self.loader_test
        
        # 2. 评估逻辑
        if model is None:
            config_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
            model_path = 'EXPS/BASE_CIFAR/full_boosted/noniid1000/eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
            model = load_model_eval(self.args, model_path=model_path, config_path=config_path)
            
        model.to(0)
        model.eval()
        
        exits_num = len(model.config.exits)
        correct = [0] * exits_num
        total = 0
        
        with torch.no_grad():
            from dataset.cifar100_dataset import CIFARClassificationDataset
            for data in test_loader:
                pixel_values = data['pixel_values'].to(0)
                labels = data['labels'].view(-1).to(0)
                
                # 根据 dataset 类型转换数据 (参考 baseHFL.adapt_batch)
                if 'cifar' in self.args.dataset:
                    pixel_values = CIFARClassificationDataset.transform_for_vit(pixel_values)
                
                logits_list = model(pixel_values=pixel_values)
                
                for i in range(exits_num):
                    logits = logits_list[i]
                    _, predicted = torch.max(logits, 1)
                    correct[i] += (predicted == labels).sum().item()
                
                total += labels.size(0)
        
        accs = [100 * c / total for c in correct]
        print(f"Evaluation Accuracies: {accs}")
        return accs

if __name__ == '__main__':
    tester = Test()
    # tester.test_slimmable_vit()
    # tester.eval()

