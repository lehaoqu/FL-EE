import argparse
import importlib
import sys


def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Method Setting ======
    parser.add_argument('alg', type=str, default='fedavg')
    # ===== train-test mismatch Policy =====
    parser.add_argument('policy', type=str, default='base', help="early exit train-test mismatch policy")
    
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--config_path', type=str, default='models/facebook/deit-tiny-patch16-224')

    # ===== Training Setting =====
    parser.add_argument('--ft', type=str, default='full', help="full | classifer | lora")
    parser.add_argument('--load_path', type=str, default='', help="pretrained model load path")
    parser.add_argument('--seed', type=int, default=1117, help="seed")
    parser.add_argument('--total_num', type=int, default=120, help="Total clients num")
    parser.add_argument('--sr', type=float, default=0.3, help="Clients sample rate")
    parser.add_argument('--suffix', type=str, default='default', help="Suffix for file")
    parser.add_argument('--device', type=int, default=0, help="Device to use")

    parser.add_argument('--rnd', type=int, default=500, help="Communication rounds")
    parser.add_argument('--bs', type=int, default=32, help="Batch size")
    parser.add_argument('--epoch', type=int, default=1, help="Epoch num")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Exponential decay of learning rate")
    parser.add_argument('--optim', type=str, default='sgd', help="optimizer")

    parser.add_argument('--valid_gap', type=int, default=10, help='Rounds between two valid phases')

    # ===== Clients Heterogeneous Setting =====
    parser.add_argument('--eq_ratios', default=(3/12, 3/12, 3/12, 3/12), type=float, nargs='+', help='device\'s size ratio')
    parser.add_argument('--eq_depths', default=(3, 6, 9, 12), type=int, nargs='+', help='device\'s depth')
    
    
    # ===== Eval Setting =====
    parser.add_argument('--if_mode', type=str, default='all', help='Mode of inference')
    parser.add_argument('--cosine', action='store_true', help='cosin')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='the ratio of valid for train_dataset')
    parser.add_argument('--eval_models_dir', type=str, default='script/0818-1e-1', help='dict need to evaled. config.json and model.pkl')

    # ===== Method Specific Setting =====
    spec_alg = sys.argv[1]
    spec_policy = sys.argv[2]
    trainer_module = importlib.import_module(f'trainer.alg.{spec_alg}')
    policy_module = importlib.import_module(f'trainer.policy.{spec_policy}')
    parser = trainer_module.add_args(parser)
    parser = policy_module.add_args(parser)
    return parser.parse_args()
