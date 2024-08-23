import argparse
import importlib
import sys

class Test():
    def __init__(self, args) -> None:
        self.alg = args.alg
    def p(self):
        print(self.alg)

def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('--alg', type=str, default='fedavg')
    return parser.parse_args()

args = args_parser()
t1 = Test(args)
t1.p()
args.alg = 'ff'
t1.p()