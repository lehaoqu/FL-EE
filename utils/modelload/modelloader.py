from utils.modelload.model import *

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

def load_model(args, model_depth=None):
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
            eq_config = pre_model.config
            eq_config.num_hidden_layers = model_depth
            eq_exit_config = ViTExitConfig(eq_config, num_labels=100) 
            eq_exit_config.output_hidden_states = True
            model = ViTExitForImageClassification(eq_exit_config)
            model.load_state_dict(pre_model.state_dict(), strict=False)
            # for (name, param) in model.named_parameters():
            #     print(name)
                
    else:
        exit('Error: unrecognized model')
    return model