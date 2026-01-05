import importlib
import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import random
import numpy as np
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.speechcmd_dataset import SPEEDCMDSClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.imagenet_dataset import TinyImageNetClassificationDataset
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model, load_model_eval
from dataset.utils import dataset_utils
from utils.modelload.modelloader import CIFAR100, SVHN, IMAGENET, SPEECHCMDS, GLUE


def load_split_pkls(prefix, total_num):
	arrays_x = []
	arrays_y = []
	for i in range(total_num):
		path = f"{prefix}{i}.pkl"
		if not os.path.exists(path):
			raise FileNotFoundError(f"Split file not found: {path}")
		dic = dataset_utils.load_pkl(path)
		# support multiple pickle formats (str keys 'x','y' or bytes keys b'data', b'fine_labels')
		x = None
		y = None
		if isinstance(dic, dict):
			if 'x' in dic:
				x = dic.get('x')
			elif b'x' in dic:
				x = dic.get(b'x')
			elif 'data' in dic:
				x = dic.get('data')
			elif b'data' in dic:
				x = dic.get(b'data')

			if 'y' in dic:
				y = dic.get('y')
			elif b'y' in dic:
				y = dic.get(b'y')
			elif 'fine_labels' in dic:
				y = dic.get('fine_labels')
			elif b'fine_labels' in dic:
				y = dic.get(b'fine_labels')
		else:
			x = dic
			y = None
		if y is None:
			raise RuntimeError(f"No label array found in {path}")
		arrays_x.append(x)
		arrays_y.append(y)

	X = np.concatenate(arrays_x, axis=0)
	Y = np.concatenate(arrays_y, axis=0)
	return X, Y


def prepare_dataloader(X, Y, batch_size, device):
	# X may be (N,3072) or (N,3,32,32) or (N,32,32,3)
	X = np.array(X)
	if X.ndim == 2:
		# flattened
		X = X.reshape(-1, 3, 32, 32)
	elif X.ndim == 4 and X.shape[-1] == 3:
		# (N,H,W,C) -> (N,C,H,W)
		X = X.transpose(0, 3, 1, 2)

	X = torch.from_numpy(X).float() / 255.0
	Y = torch.from_numpy(np.array(Y)).long()


	# vectorized batch transform using interpolate + broadcast normalization
	mean = torch.tensor((0.5070751592371323, 0.48654887331495095, 0.4409178433670343)).view(1,3,1,1)
	std = torch.tensor((0.2673342858792401, 0.2564384629170883, 0.27615047132568404)).view(1,3,1,1)

	dataset = TensorDataset(X, Y)

	def collate_fn(batch):
		imgs = torch.stack([b[0] for b in batch], dim=0)
		# imgs: (B,C,H,W) in [0,1]
		imgs = F.interpolate(imgs, size=(224,224), mode='bilinear', align_corners=False)
		imgs = (imgs - mean) / std
		labels = torch.stack([b[1] for b in batch], dim=0)
		return imgs.to(device), labels.to(device)

	# use a small number of workers to parallelize data loading without overwhelming CPU
	num_workers = min(4, max(0, (os.cpu_count() or 1) // 2))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
	return loader


def adapt_batch(args, data):
	batch = {}
	for key in data.keys():
		batch[key] = data[key].to(args.device)
		if key == 'pixel_values':
			if 'cifar' in args.dataset:
				batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
			elif 'imagenet' in args.dataset:
				batch[key] = TinyImageNetClassificationDataset.transform_for_vit(batch[key])
			elif 'speechcmds' in args.dataset:
				batch[key] = SPEEDCMDSClassificationDataset.transform_for_vit(batch[key])
			else:
				batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
	label = batch['labels'].view(-1)
	return batch, label


def distill(args, teacher, student, train_loader, valid_loader, device, epochs=1, lr=1e-3, temperature=4.0, alpha=0.0, save_log=None):
	teacher.to(device).eval()
	student.to(device).train()

	optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, student.parameters()), lr=lr, momentum=0.9)
	ce_loss = nn.CrossEntropyLoss()
	kl_loss = nn.KLDivLoss(reduction='batchmean')
	pbar = tqdm(range(epochs), desc='Distillation', unit='epoch')
	save_log = open(save_log, 'w') if save_log is not None else None

	for epoch in pbar:
		total_loss = 0.0
		total_samples = 0
		for data in train_loader:
			loss = 0.0
			optimizer.zero_grad()
			batch, label = adapt_batch(args, data)
			with torch.no_grad():
				t_logits = teacher(**batch)
			s_logits = student(**batch)
			for idx, s_exit_logits in enumerate(s_logits):
				loss_ce = ce_loss(s_exit_logits, label)
				t_soft = torch.softmax(t_logits[idx] / temperature, dim=1)
				s_logsoft = torch.log_softmax(s_exit_logits / temperature, dim=1)
				loss_kd = kl_loss(s_logsoft, t_soft) * (temperature ** 2)
				loss += alpha * loss_ce + (1 - alpha) * loss_kd
			loss.backward()
			optimizer.step()

			bs = label.size(0)
			total_loss += loss.item() * bs
			total_samples += bs

		avg_loss = total_loss / (total_samples + 1e-12)
		pbar.set_description(f'Distillation Loss: {avg_loss:.4f}')
		save_log.write(f"Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}\n") if save_log is not None else None
		# print(f"Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}")

	return student


def load_weight_variant_model(args, teacher_model, exits=None):
	import copy
	from utils.modelload.modelloader import crop_tensor_dimensions
	dataset_arg = args.dataset
	model_arg = args.model
	depth = args.student_depth
	scales = args.scales

	based_model = importlib.import_module(f'utils.modelload.{model_arg}')
	num_labels = 100 if CIFAR100 in dataset_arg else 10 if SVHN in dataset_arg else 200 if IMAGENET in dataset_arg else 35 if SPEECHCMDS in dataset_arg else 2    
	variant_models = {}

	for scale in scales:
		variant_config = copy.deepcopy(teacher_model.config)
		variant_config.num_hidden_layers = depth
		# print(f'Applying scale: {scale}')
		variant_config.hidden_size = int(variant_config.hidden_size * scale // variant_config.num_attention_heads * variant_config.num_attention_heads)
		variant_config.intermediate_size = int(variant_config.intermediate_size * scale // variant_config.num_attention_heads * variant_config.num_attention_heads)
		variant_exit_config = based_model.ExitConfig(variant_config, num_labels=num_labels, exits=exits, policy=teacher_model.config.policy, alg=teacher_model.config.alg, blocks=teacher_model.config.blocks) 
		# print(teacher_model.config.policy)
		model = based_model.ExitModel(variant_exit_config)
		
		origin_target = {teacher_model.config.hidden_size: variant_config.hidden_size, teacher_model.config.intermediate_size: variant_config.intermediate_size}
		# print(f'scale: {scale}, width: {origin_target}')
		
		new_state_dict = {}
		for name, param in model.named_parameters():
			if name in teacher_model.state_dict().keys():
				origin_tensor = teacher_model.state_dict()[name]
				if 'bert.embeddings.position' in name: prune_tensor = crop_tensor_dimensions(origin_tensor, {teacher_model.config.hidden_size: variant_config.hidden_size})
				else: prune_tensor = crop_tensor_dimensions(origin_tensor, origin_target)
				param = prune_tensor.clone()
			new_state_dict[name] = param
		model.load_state_dict(new_state_dict)
		variant_models[scale] = model
	return variant_models
	

def main():
	parser = argparse.ArgumentParser(description='Distill teacher .pth into student model')
	parser.add_argument('--teachers_dir', required=True, help='Dir for teachers .pth file')
	parser.add_argument('--valid_prefix', required=False, help='Prefix for valid split files')
	parser.add_argument('--total_num', type=int, default=100, help='Number of split files/clients')
	parser.add_argument('--model', type=str, default='vit', help='Model name (matches utils.modelload module)')
	parser.add_argument('--dataset', type=str, default='cifar100_noniid1000')
	parser.add_argument('--epochs', type=int, default=500)
	parser.add_argument('--bs', type=int, default=32)
	parser.add_argument('--lr', type=float, default=0.05)
	parser.add_argument('--device', type=str, default='0')

	# limit PyTorch thread usage to avoid excessive CPU spin
	max_threads = min(4, max(1, (os.cpu_count() or 1)))
	torch.set_num_threads(max_threads)
	os.environ.setdefault('OMP_NUM_THREADS', str(max_threads))
	os.environ.setdefault('MKL_NUM_THREADS', str(max_threads))
	
	args = parser.parse_args()
	args.scales = [0.33, 0.67]
	args.device = torch.device('cuda:' + args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')
	
	if CIFAR100 in args.dataset or SVHN in args.dataset or SPEECHCMDS in args.dataset:
		args.train_prefix = f'./dataset/{args.dataset}/train/'
	elif GLUE in args.dataset:
		args.train_prefix = f'./dataset/glue/{args.dataset}/train/'

	teachers_dir = args.teachers_dir
	file_names = os.listdir(teachers_dir)
	model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f and '.png' not in f]))
	model_paths = [f'./{teachers_dir}/{model_name}' for model_name in model_names]
	for teacher_path in model_paths:
		if 'G' in teacher_path or 'loss' in teacher_path or 'acc' in teacher_path or 'distance' in teacher_path or 'budget' in teacher_path:
			continue
		# TODO
		if 'reefl' in teacher_path:
			continue
		print('Processing teacher model:', teacher_path)
		args.teacher_pth = teacher_path + '.pth'
		args.teacher_config = teacher_path + '.json'
		args.output_prefix = teacher_path + '_variants/variant'
		teacher_distillation(args)


def teacher_distillation(args):
	# build a lightweight args namespace for model loaders
	device = args.device

	class A: pass
	margs = A()
	margs.model = args.model
	margs.dataset = args.dataset
	margs.ft = 'full'
	margs.load_path = ''
	margs.policy = 'base'
	margs.alg = 'depthfl'
	margs.blocks = 12

	teacher_config = args.teacher_config
	teacher_pth = args.teacher_pth
	print('Loading teacher model from', teacher_pth, 'with config', teacher_config)
	teacher = load_model_eval(margs, teacher_pth, config_path=teacher_config)

	# distill students with varying depths and widths
	for student_depth in (3,6,9,12):
		print('Initializing student model (depth=', student_depth,')')
		exits = (2,5,8,11) if student_depth == 12 else (2,5,8) if student_depth == 9 else (2,5) if student_depth == 6 else (2,)
		args.student_depth = student_depth
		variant_students = load_weight_variant_model(args, teacher_model=teacher, exits=exits)

		# Distill each variant student in widths
		for student_width, student in variant_students.items():
			print(f'Distilling student with depth {student_depth} & width {student_width} | prepare dataset...')
			wandb.init(project=f"Variant_Distillation_{args.dataset}", name=f"{teacher_pth.split('/')[-1].split('_')[0]}_{teacher_pth.split('/')[3]}_depth{student_depth}_width{student_width}")
			dataset_idx = 0 if student_depth == 3 else 25 if student_depth == 6 else 50 if student_depth == 9 else 75
			dataset_train, loader_train = load_dataset_loader(args=args, file_name='train', id=dataset_idx)
			dataset_valid, loader_valid = load_dataset_loader(args=args, file_name='valid', id=dataset_idx)

			os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
			save_log = args.output_prefix + f'_depth{student_depth}_width{student_width}.txt'
			save_scale_config = args.output_prefix + f'_depth{student_depth}_width{student_width}.json'
			with open(save_scale_config, 'w', encoding='utf-8') as f:
				json.dump(student.config.to_dict(), f, ensure_ascii=False, indent=4)

			# print(student.config)
			student = distill(args, teacher, student, loader_train, loader_valid, device, epochs=args.epochs, lr=args.lr, save_log=save_log)
			
			save_scale_pth = args.output_prefix + f'_depth{student_depth}_width{student_width}.pth'
			torch.save(student.state_dict(), save_scale_pth)
			print('Saved distilled student to', save_scale_pth)


if __name__ == '__main__':
	seed = 1117
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	main()