import torch
import argparse
import random
from trainer.generator.generator import Generator_CIFAR



def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('--alg', type=str, default='fedavg')
    return parser.parse_args()

args = args_parser()
args.device = 0

y_distribute = [0.1, 0.2, 0.3, 0.4]
y_input = torch.tensor(random.choices(range(len(y_distribute)), weights=y_distribute, k=32), dtype=torch.float)
print(y_input.shape)