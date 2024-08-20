import copy
import json

from utils.modelload.model import *
from utils.train_utils import *

MNIST = 'mnist'
CIFAR10 = 'cifar10'
CIFAR100 = 'cifar100'

MLP = 'mlp'
CNN = 'cnn'
VIT = 'vit'

def check_class_num(dataset):
    return 100 if CIFAR100 in dataset \
        else 10 if CIFAR10 in dataset \
        else 10 if MNIST in dataset \
        else -1

def load_model(args, model_depth=None, is_scalefl=False):
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
    if CIFAR100 in dataset_arg:
        if VIT in model_arg:
            config_path = args.config_path
            pre_model = ViTForImageClassification.from_pretrained(pretrained_model_name_or_path=config_path)
            eq_config = copy.deepcopy(pre_model.config)
            
            if is_scalefl:
                depth = min(12, model_depth+1)
                scale = model_depth / depth
                eq_config.num_hidden_layers = depth
                eq_config.hidden_size = int(eq_config.hidden_size * scale // eq_config.num_attention_heads * eq_config.num_attention_heads)
                eq_config.intermediate_size = int(eq_config.intermediate_size * scale // eq_config.num_attention_heads * eq_config.num_attention_heads)
                eq_exit_config = ViTExitConfig(eq_config, num_labels=100, exits=(3,6,9,11)) 
                model = ViTExitForImageClassification(eq_exit_config)
                
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
                eq_exit_config = ViTExitConfig(eq_config, num_labels=100, exits=(2,5,8,11)) 
                model = ViTExitForImageClassification(eq_exit_config)
                model.load_state_dict(pre_model.state_dict(), strict=False)

            # tensors = model.parameters_to_tensor((2,5,8,11), is_split=True)
            # print(len(tensors))
            # for tensor in tensors:
            #     print(tensor.shape)
                
            # print(model.parameters_to_tensor().shape)
            # for name, param in model.named_parameters():
            #     print(name, param.shape)
        
    else:
        exit('Error: unrecognized model')
    return model

def load_model_eval(args, model_path, config_path=None):
    model_arg = args.model
    dataset_arg = args.dataset
    if CIFAR100 in dataset_arg:
        if VIT in model_arg:            
            exit_config = ViTConfig.from_pretrained(pretrained_model_name_or_path=config_path)
            model = ViTExitForImageClassification(config=exit_config)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            print('model load compeleted')
    
    else:
        exit('Error: unrecognized model')
    
    return model
