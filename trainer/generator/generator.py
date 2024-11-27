import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.cifar100_dataset import CIFARClassificationDataset
from utils.modelload.model import BaseModule

CLASSES = {'imagenet':200, 'svhn':10, 'cifar100_noniid1000': 100, 'cifar100_noniid1': 100, 'cifar100_noniid0.1': 100, 'cifar100-224-d03-1200': 100, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'qnli': 2, 'rte': 2, 'wnli': 2}

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


class Generator_LATENT(BaseModule):
    def __init__(self, args=None, embedding=True):
        super(Generator_LATENT, self).__init__()
        self.args = args
        self.device = args.device if args is not None else 0
        self.embedding = embedding
        # TODO latent_dim n_class will change in glue and cifar
        if 'cifar' in args.dataset or 'svhn' in args.dataset or 'imagenet' in args.dataset:
            if 'tiny' in args.config_path:
                self.hidden_dim, self.token_num, self.hidden_rs, self.n_class, self.noise_dim = 1000, 197, 192, CLASSES[args.dataset], args.noise
            elif 'small' in args.config_path:
                self.hidden_dim, self.token_num, self.hidden_rs, self.n_class, self.noise_dim = 1000, 197, 384, CLASSES[args.dataset], args.noise
        else:
            self.hidden_dim, self.token_num, self.hidden_rs, self.n_class, self.noise_dim = 1000, 128, 128, CLASSES[args.dataset], args.noise
        self.latent_dim = self.token_num * self.hidden_rs
        
        input_dim = 3 * self.noise_dim  if self.args.diff_generator else 2 * self.noise_dim 
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()
    
    def init_loss_fn(self):
        self.crossentropy_loss=nn.CrossEntropyLoss() # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        
    def build_network(self):
        # self.embedding_layer_diff = nn.Embedding(self.n_diff, self.noise_dim)
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        if self.args.diff_generator:
            self.diff_laryer = nn.Linear(len(self.args.exits), self.noise_dim)
        ## == FC ==
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i+1]
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        
        # == Representation Layer ==
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        
    def forward(self, labels, eps, exits_diff=None):
        if isinstance(labels, tuple): labels = labels[0]
        batch_size = labels.shape[0]
        
        # diff_embedding = self.embedding_layer_diff(diffs)
        if self.embedding:
            y_input = self.embedding_layer(labels)
        else:
            y_input = F.one_hot(labels, num_classes=self.n_class).float()
        if self.args.diff_generator: z = torch.cat((eps, y_input, self.diff_laryer(exits_diff)), dim=1)
        else: z = torch.cat((eps, y_input), dim=1)
        
        # == FC Layers ==
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        return z.view(batch_size, self.token_num, self.hidden_rs)


    def statistic_loss(self, gen_latent, train_mean, train_std):
        # g_mean = gen_latent.mean([0,2], keepdim=True)
        # g_std = gen_latent.std([0,2], keepdim=True)
        
        # mean_loss = torch.mean((g_mean - train_mean) **2)
        # std_loss = torch.mean((g_std - train_std) **2)
        # loss = mean_loss + std_loss
        loss = torch.mean(torch.mean(torch.abs(torch.mean(gen_latent, dim=0)-train_mean), dim=1))
        return loss


class Generator_CIFAR(BaseModule):
    def __init__(self, args=None, dataset='cifar100_dataset'):
        hidden_channel, output_channel, img_size, n_class, noise_dim, n_diff = 64, 3, 32, 100, 100, 10
        super(Generator_CIFAR, self).__init__()
        
        self.noise_dim = noise_dim
        self.diversity_loss = DiversityLoss(metric='l1')
        
        self.device = args.device if args is not None else 0

        self.init_size = img_size//4
        
        self.embedding_layer = nn.Embedding(n_class, noise_dim)
        self.embedding_layer_diff = nn.Embedding(n_diff, noise_dim)
        self.l1 = nn.Sequential(nn.Linear(noise_dim*3, hidden_channel*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(hidden_channel*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(hidden_channel*2, hidden_channel*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channel*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(hidden_channel*2, hidden_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channel, output_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            # nn.BatchNorm2d(output_channel, affine=False) 
        )

    def forward(self, diffs, labels, noise, raw=False):
        if isinstance(labels, tuple): labels = labels[0]
        batch_size = labels.shape[0]
        y_embedding = self.embedding_layer(labels)
        diff_embedding = self.embedding_layer_diff(diffs)
        
        z = torch.cat((noise, y_embedding, diff_embedding), dim=-1)
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img if raw else CIFARClassificationDataset.generator_transform_tensor(images=img)


    def statistic_loss(self, g_images, train_mean, train_std):
        g_mean = g_images.mean([0,2,3], keepdim=True)
        g_std = g_images.std([0,2,3], keepdim=True)
        
        mean_loss = torch.mean((g_mean - train_mean) **2)
        std_loss = torch.mean((g_std - train_std) **2)
        loss = mean_loss + std_loss
        return loss