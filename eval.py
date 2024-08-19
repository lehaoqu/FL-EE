import torch

from tqdm import tqdm

from utils.options import args_parser
from dataset.utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model_eval



class Eval():
    def __init__(self, args):
        self.args = args
        self.if_mode = args.if_mode
        self.device = args.device
        args.valid_ratio = 0.8
        self.valid_dataloader = load_dataset_loader(args=args, is_valid=True)
        self.test_dataloader = load_dataset_loader(args=args, is_valid=False)
        
        
    
    def eval(self, model_path, config_path):
        self.model = load_model_eval(self.args, model_path, config_path)
        self.exits_num = len(self.model.config.exits)
        self.model.to(self.device)
        self.model.eval()
        if self.args.if_mode == 'anytime':
            acc_list = self.anytime()
            print(acc_list)
        else:
            self.budgeted()
        
    
    def anytime(self,):
        correct_list = [0 for _ in range(self.exits_num)]
        total_list = [0 for _ in range(self.exits_num)]
        
        with torch.no_grad():
            for data in tqdm(self.test_dataloader):
                batch = {}
                for key in data.keys():
                    batch[key] = data[key].to(self.device)
                y = batch['labels']
                exits_logits = self.model(**batch)
                for i, exit_logits in enumerate(exits_logits):
                    _, predicted = torch.max(exit_logits, 1)
                    total_list[i] += y.size(0)
                    correct_list[i] += (predicted == y).sum().item()
        
        acc_list = [100 * correct_list[i] / total_list[i] for i in range(self.exits_num)]
        return acc_list
        
    
    def budgeted(self,):
        
        
        # for p in range(1, 40):
        #     print("*********************")
        #     _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        #     n_blocks = args.nBlocks * len(args.scale_list) if args.arch == 'ranet' else args.nBlocks
        #     probs = torch.exp(torch.log(_p) * torch.range(1, n_blocks))
        #     probs /= probs.sum()
        #     acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops)
        #     acc_test, exp_flops = tester.dynamic_eval_with_threshold(test_pred, test_target, flops, T)
        
        # for torch.no_grad():
        pass
            
    
if __name__ == '__main__':
    args = args_parser()
    eval = Eval(args=args)
    path = '/data/qvlehao/FL-EE/exps/depthfl/depthfl_cifar100-224-d03_vit_4c_1E_lr0.1'
    eval.eval(path+'.pth', path+'.json')