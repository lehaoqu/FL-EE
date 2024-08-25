import copy
import json
import importlib

from utils.modelload.model import *
from utils.train_utils import *

MNIST = 'mnist'
CIFAR10 = 'cifar10'
CIFAR100 = 'cifar100'
GLUE = ['sst2', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli']

MLP = 'mlp'
CNN = 'cnn'
VIT = 'vit'
BERT = 'bert'

def check_class_num(dataset):
    return 100 if CIFAR100 in dataset \
        else 10 if CIFAR10 in dataset \
        else 10 if MNIST in dataset \
        else -1

def load_model(args, model_depth=None, is_scalefl=False, exits=None):
    model_arg = args.model
    dataset_arg = args.dataset
    class_num = check_class_num(dataset_arg)

    if MNIST in dataset_arg:
        if MLP in model_arg:
            model = MLP(args=args, dim_in=784, dim_hidden=256, dim_out=class_num)
        elif CNN in model_arg:
            model = CNNMnist(args=args, dim_out=class_num)
    if CIFAR10 in dataset_arg:
        if CNN in model_arg:
            model = CNNCifar(args=args, dim_out=class_num)
    if CIFAR100 in dataset_arg or dataset_arg in GLUE:
        
        based_model = importlib.import_module(f'utils.modelload.{model_arg}')
        
        config_path = args.config_path
        pre_model = based_model.Model.from_pretrained(pretrained_model_name_or_path=config_path)
        eq_config = copy.deepcopy(pre_model.config)
        
        num_labels = 100 if CIFAR100 in dataset_arg else 2
        
        if is_scalefl:
            depth = min(12, model_depth+1)
            scale = model_depth / depth
            eq_config.num_hidden_layers = depth
            eq_config.hidden_size = int(eq_config.hidden_size * scale // eq_config.num_attention_heads * eq_config.num_attention_heads)
            eq_config.intermediate_size = int(eq_config.intermediate_size * scale // eq_config.num_attention_heads * eq_config.num_attention_heads)
            eq_exit_config = based_model.ExitConfig(eq_config, num_labels=num_labels, exits=exits, policy=args.policy, alg=args.alg) 
            model = based_model.ExitModel(eq_exit_config)
            
            origin_target = {pre_model.config.hidden_size: eq_config.hidden_size, pre_model.config.intermediate_size: eq_config.intermediate_size}
            print(f'scale: {scale}, width: {origin_target}')
            new_state_dict = {}
            for name, param in model.named_parameters():
                if name in pre_model.state_dict().keys():
                    origin_tensor = pre_model.state_dict()[name]
                    prune_tensor = crop_tensor_dimensions(origin_tensor, origin_target)
                    param = prune_tensor.clone()
                new_state_dict[name] = param
            model.load_state_dict(new_state_dict)
            
        else:
            eq_config.num_hidden_layers = model_depth
            exits = (2,5,8,11)
            
            eq_exit_config = based_model.ExitConfig(eq_config, num_labels=num_labels, exits=exits, policy=args.policy, alg=args.alg) 
            model = based_model.ExitModel(eq_exit_config)
            model.load_state_dict(pre_model.state_dict(), strict=False)

            # tensors = model.parameters_to_tensor((2,5,8,11), is_split=True)
            # print(len(tensors))
            # for tensor in tensors:
            #     print(tensor.shape)
            
            # print(model.parameters_to_tensor().shape)
            # for name, param in model.named_parameters():
            #     print(name, param.shape)
            
            # exit(0)
        
        # if model_arg == 'bert':
        #     model = model.half()

    else:
        exit('Error: unrecognized model')
    return model

def load_model_eval(args, model_path, config_path=None):
    model_arg = args.model
    dataset_arg = args.dataset
    if CIFAR100 in dataset_arg or dataset_arg in GLUE:
        based_model = importlib.import_module(f'utils.modelload.{model_arg}')
           
        exit_config = based_model.Config.from_pretrained(pretrained_model_name_or_path=config_path)
        model = based_model.ExitModel(config=exit_config)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print('model load compeleted')
    
    else:
        exit('Error: unrecognized model')
    
    return model
