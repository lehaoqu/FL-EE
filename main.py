import importlib
import sys
import numpy as np
import os
import copy
import json
import torch
import numpy as np
import random

from utils.options import args_parser
from utils.dataprocess import DataProcessor
from tqdm import tqdm
from utils.modelload.modelloader import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FedSim:
    def __init__(self, args):
        self.args = args

        # === load trainer ===
        trainer_module = importlib.import_module(f'trainer.alg.{args.alg}') if args.alg != 'fl' else importlib.import_module(f'trainer.alg.exclusivefl')

        # === init other config ===
        self.acc_processor = DataProcessor()
        self.acc_shift_processor = DataProcessor()

        if not os.path.exists(f'./{args.suffix}'):
            os.makedirs(f'./{args.suffix}')
        
        if args.alg == 'fl':
            args.eq_ratios = (0,0,0,1)

        self.model_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.pth'
        self.generator_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}_G.pth'
        output_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.txt'   
        self.acc_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}_acc.json'
        self.loss_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}_loss.json'
        self.config_save_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                    f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}.json'    
                    
        self.distance_path = f'./{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
            f'{args.total_num}c_{args.epoch}E_lr{args.optim}{args.lr}_{args.policy}_distance.json'

        self.output = open(output_path, 'a')
        args.output = self.output

        if os.path.exists(self.loss_path):
            return

        for arg in vars(args):
            self.output.write(f"{arg}: {getattr(args, arg)}\n")
        self.output.flush()

        # === init pre-trainde model ===
        ratios = ()
        for i in range(len(args.eq_ratios)):
            ratios += (sum(args.eq_ratios[:i+1]),)
        ranges_to_gropus = {int(args.total_num * ratio)-1 : group for (group, ratio) in enumerate(ratios)}
        # print(ranges_to_gropus)
        eq_model = {}
        
        # == for scalefl & heterofl ==
        # exits = (i-1 for i in args.eq_depths)
        # if args.alg == 'scalefl': exits = (i+1 for _, i in enumerate(exits) if _!=len(exits-1))
        # exits = (2,5,8,11) if args.alg != 'scalefl' else (3,6,9,11)
        if args.dataset == 'svhn' and args.policy == 'base' and args.ft == 'full':
            exits, blocks = (3,5,8,11), (3,5,8,11)
        else:
            exits = tuple(i-1 for i in args.eq_depths) if args.alg != 'scalefl' else tuple(i if i != max(args.eq_depths) else i-1 for i in args.eq_depths)
            blocks = tuple(i-1 for i in args.eq_depths) if args.alg != 'scalefl' else tuple(i if i != max(args.eq_depths) else i-1 for i in args.eq_depths)
        args.blocks = blocks
        if args.alg != 'heterofl': eq_exits = {eq_depth: exits[:int((args.eq_depths.index(eq_depth)+1)*len(exits)/len(args.eq_depths))] for eq_depth in args.eq_depths}
        else: eq_exits = {eq_depth: exits for eq_depth in args.eq_depths}
        
        if args.multi_exit:
            eq_exits = {eq_depth: tuple(_ for _ in range(max(eq_exits[eq_depth])+1)) for eq_depth in args.eq_depths}
        
        for depth in args.eq_depths:
            eq_model[depth] = load_model(args, model_depth=depth, is_scalefl=(args.alg == 'scalefl' or args.alg == 'heterofl'), exits=eq_exits[depth])
        
        # == for scalefl & heterofl ==
        largest_model = eq_model[max(args.eq_depths)]
        args.origin_width = [largest_model.config.hidden_size, largest_model.config.intermediate_size]
        
        # === init clients & server ===
        self.clients = []
        for idx in tqdm(range(args.total_num), desc='Client compeleted'):
            for end_of_range in sorted(ranges_to_gropus, reverse=False):
                if idx <= end_of_range: 
                    depth = args.eq_depths[ranges_to_gropus[end_of_range]]
                    break
            client_exits = eq_exits[depth]
            self.clients.append(trainer_module.Client(idx, args, None, copy.deepcopy(eq_model[depth]), depth, client_exits))
            # print(f'client {idx} compeleted')
        self.server = trainer_module.Server(0, args, None, self.clients, copy.deepcopy(eq_model), copy.deepcopy(eq_model[max(args.eq_depths)]), eq_exits)

    def simulate(self):
        
        if os.path.exists(self.loss_path):
            return
        
        # == save global model's config if model is transformer type ==
        with open(self.config_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.server.global_model.config.to_dict(), f, ensure_ascii=False, indent=4)
        
        valid_GAP = self.args.valid_gap if args.dataset not in ('mrpc', 'rte') else 1
        best_acc = 0.0
        best_rnd = 0
        valid_acc = []
        losses = []
        try:
            for rnd in tqdm(range(self.server.total_round), desc='Communication Round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== valid =====================
                if rnd > 200:
                    valid_GAP = 2 if self.args.alg != 'darkflpg' else 1
                if rnd % valid_GAP:
                    continue

                ret_dict = self.server.valid_all()
                self.acc_processor.append(ret_dict['acc'])
                if ret_dict['acc'] > best_acc:
                    best_acc = ret_dict['acc']
                    best_rnd = rnd
                    self.server.save_model(self.model_save_path, self.generator_save_path)
                    best_model = copy.deepcopy(self.server.global_model)
                    best_model.to(self.args.device)

                self.output.write(f'========== Round {rnd} ==========\n')
                # print(f'========== Round {rnd} ==========\n')
                acc_exits = [f"{num:.2f}" for num in ret_dict['acc_exits']]
                self.output.write(f"server, accuracy: {ret_dict['acc']:.2f}, exits:{acc_exits} loss: {ret_dict['loss']:.2f}\n")
                if rnd % 10 == 0:
                    valid_acc.append(ret_dict['acc'])
                    losses.append(ret_dict['loss'])
                # self.output.write('wall clock time: %.2f seconds\n' % self.server.wall_clock_time)
                self.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            acc_list = self.acc_processor.data
            # np.save(f'./{self.args.suffix}/{self.args.alg}_{self.args.dataset}'
            #         f'_{self.args.model}_{self.args.total_num}c_{self.args.epoch}E_lr{args.lr}_{args.policy}.npy',
            #         np.array(acc_list))
            avg_count = 2
            acc_avg = np.mean(acc_list[-avg_count:]).item()
            acc_std = ret_dict['std']
            acc_max = np.max(acc_list).item()

            self.output.write('==========Summary==========\n')

            self.output.write('server, max accuracy: %.2f\n' % acc_max)
            self.output.write('server, final accuracy: %.2f +- %.2f\n' % (acc_avg, acc_std))

            if self.args.alg == 'darkflpg':
                self.server.save_distance(self.distance_path)
            
            self.server.calc_logits(best_model)
            self.server.anytime(self.output)
            with open(self.acc_path, 'w') as file:
                json.dump(valid_acc, file)
            with open(self.loss_path, 'w') as file:
                json.dump(losses, file)
            # self.res_output.write(f'{self.args.alg}, best_rnd: {best_rnd}, acc: {acc_avg:.2f}+-{acc_std:.2f}\n')


if __name__ == '__main__':
    args = args_parser()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    fed = FedSim(args=args)
    fed.simulate()
