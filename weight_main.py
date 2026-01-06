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
import warnings

from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.speechcmd_dataset import SPEEDCMDSClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress specific Pydantic warnings about unsupported Field attributes
# These originate from pydantic._internal._generate_schema when Field(...)
# is used in a type alias or union where 'repr'/'frozen' have no effect.
warnings.filterwarnings(
	"ignore",
	module=r"pydantic._internal._generate_schema"
)

# Suppress SWIG-related DeprecationWarnings from some compiled extensions
warnings.filterwarnings(
	"ignore",
	message=r"builtin type SwigPyPacked has no __module__ attribute",
	category=DeprecationWarning,
)
warnings.filterwarnings(
	"ignore",
	message=r"builtin type SwigPyObject has no __module__ attribute",
	category=DeprecationWarning,
)
warnings.filterwarnings(
	"ignore",
	message=r"builtin type swigvarlink has no __module__ attribute",
	category=DeprecationWarning,
)

from dataset.imagenet_dataset import TinyImageNetClassificationDataset
from utils.dataloader_utils import load_dataset_loader
from utils.modelload.modelloader import load_model, load_model_eval
from dataset.utils import dataset_utils
from utils.modelload.modelloader import CIFAR100, SVHN, IMAGENET, SPEECHCMDS, GLUE


def compare_model_instances(model_a, model_b):
    # normalize and move to cpu for safe comparison
    def _norm_state(sd):
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        return {k[len("module."): ] if k.startswith("module.") else k: v.cpu() for k, v in sd.items()}

    sa = _norm_state(model_a.state_dict())
    sb = _norm_state(model_b.state_dict())

    # if set(sa.keys()) != set(sb.keys()):
    #     print(f"Keys differ: only A {set(sa)-set(sb)}, only B {set(sb)-set(sa)}")
    #     return False, f"keys differ: only A {set(sa)-set(sb)}, only B {set(sb)-set(sa)}"
    for k in sorted(sb.keys()):
        a, b = sa[k], sb[k]
        if a.shape != b.shape:
            print('shape error')
            return False, f"shape differ: {k} {a.shape} vs {b.shape}"
        if not torch.equal(a, b):
            print('value error')
            return False, f"value differ at key: {k}"
    print('========S============')
    return True, "models identical"


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

def kd_loss_func(pred, teacher, T=4.0):
	kld_loss = nn.KLDivLoss(reduction='batchmean')
	log_softmax = nn.LogSoftmax(dim=-1)
	softmax = nn.Softmax(dim=1)
	_kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
	return _kld


def distill(args, teacher, student, train_loader, valid_loader, device, epochs=1, lr=1e-3, temperature=4.0, alpha=0.0, save_log=None, wdb=None, t_policy=None, s_policy=None):
	teacher.to(device).eval()
	student.to(device).train()

	###################
	# Train
	###################
	optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, student.parameters()), lr=lr, momentum=0.9)
	ce_loss = nn.CrossEntropyLoss()
	pbar = tqdm(range(epochs), desc='Distillation', unit='epoch')
	save_log = open(save_log, 'w') if save_log is not None else None

	acc = 0.0
	for epoch in pbar:
		total_loss = 0.0
		total_samples = 0
		for data in train_loader:
			loss = 0.0
			optimizer.zero_grad()
			batch, label = adapt_batch(args, data)
			with torch.no_grad():
				t_exits_ce_loss, t_exits_logits = t_policy.train(teacher, batch, label)
				# t_logits = teacher(**batch)
			# s_logits = student(**batch)
			s_exits_ce_loss, s_exits_logits = s_policy.train(student, batch, label)
			# print(len(s_logits), len(t_logits))
			# print(s_logits[0], t_logits[0])
			# max_diff = (s_logits[0] - t_logits[0]).abs().max().item()
			# print("Max absolute difference:", max_diff)
			# print(torch.equal(s_logits[0], t_logits[0]))
			# assert torch.allclose(s_logits[0], t_logits[0]), "Teacher and student exit logits do not match in number!"
			for idx, s_exit_logits in enumerate(s_exits_logits):
				loss_ce = ce_loss(s_exit_logits, label)
				loss_kd = kd_loss_func(s_exit_logits, t_exits_logits[idx], T=temperature)
				loss += alpha * loss_ce + (1 - alpha) * loss_kd
			# print("Loss per exit:", loss.item())
			loss.backward()
			optimizer.step()

			bs = label.size(0)
			total_loss += loss.item() * bs
			total_samples += bs

		avg_loss = total_loss / (total_samples + 1e-12)
		# pbar.set_description(f'Distillation Loss: {avg_loss:.4f}')
		save_log.write(f"Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}\n") if save_log is not None else None
		save_log.flush()
		wdb.log({"distill_train_loss": avg_loss}) if wdb is not None else None
		# print(f"Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}")
		pbar.set_description(f'Distillation Loss: {avg_loss:.4f} | Valid Acc: {acc:.2f}%')

		###################
		# Validation
		###################
		if epoch % 10 != 0:
			continue
		student.eval()
		teacher.eval()
		correct = 0
		total = 0
		corrects = [0 for _ in range(args.exits_num)]
		# corrects = [0 for _ in range(4)]

		with torch.no_grad():
			for data in valid_loader:
				batch, labels = adapt_batch(args, data)
				exits_logits = student(**batch)
				exits_logits = s_policy(exits_logits)
				for i, exit_logits in enumerate(exits_logits):
					_, predicted = torch.max(exit_logits, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
					corrects[i] += (predicted == labels).sum().item()
		acc = 100.00 * correct / total
		acc_exits = [100 * c / (total/args.exits_num) for c in corrects]
		# args.metric['acc_exits'] = acc_exits
		# args.metric['acc'].append(acc)
		pbar.set_description(f'Distillation Loss: {avg_loss:.4f} | Valid Acc: {acc:.2f}%')
		acc_exits_str = ", ".join(f"{a:.2f}" for a in acc_exits)
		save_log.write(f"Epoch {epoch+1}/{epochs} valid acc: {acc:.2f}\n acc_exits: {acc_exits_str}\n") if save_log is not None else None
		save_log.flush()
		wdb.log({"distill_valid_acc": acc, "epoch": epoch+1}) if wdb is not None else None
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
			wdb = wandb.init(project=f"Variant_Distillation_{args.dataset}", name=f"{teacher_pth.split('/')[-1].split('_')[0]}_{teacher_pth.split('/')[3]}_depth{student_depth}_width{student_width}", reinit=True)
			dataset_idx = 0 if student_depth == 3 else 25 if student_depth == 6 else 50 if student_depth == 9 else 75
			dataset_train, loader_train = load_dataset_loader(args=args, file_name='train', id=dataset_idx)
			dataset_valid, loader_valid = load_dataset_loader(args=args, file_name='valid', id=dataset_idx)

			os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
			save_log = args.output_prefix + f'_depth{student_depth}_width{student_width}.txt'
			save_scale_config = args.output_prefix + f'_depth{student_depth}_width{student_width}.json'
			with open(save_scale_config, 'w', encoding='utf-8') as f:
				json.dump(student.config.to_dict(), f, ensure_ascii=False, indent=4)

			# print(student.config)
			compare_model_instances(teacher, student)
			policy_module = importlib.import_module(f'trainer.policy.{args.policy}')
			args.exits_num = 4
			t_policy = policy_module.Policy(args)
			args.exits_num = len(exits)
			s_policy = policy_module.Policy(args)
			student = distill(args, teacher, student, loader_train, loader_valid, device, epochs=args.epochs, lr=args.lr, save_log=save_log, wdb=wdb, t_policy=t_policy, s_policy=s_policy)

			save_scale_pth = args.output_prefix + f'_depth{student_depth}_width{student_width}.pth'
			torch.save(student.state_dict(), save_scale_pth)
			print('Saved distilled student to', save_scale_pth)
			wdb.finish()


def main():
	parser = argparse.ArgumentParser(description='Distill teacher .pth into student model')
	parser.add_argument('--teachers_dir', required=True, help='Dir for teachers .pth file')
	parser.add_argument('--valid_prefix', required=False, help='Prefix for valid split files')
	parser.add_argument('--total_num', type=int, default=100, help='Number of split files/clients')
	parser.add_argument('--model', type=str, default='vit', help='Model name (matches utils.modelload module)')
	parser.add_argument('--dataset', type=str, default='cifar100_noniid1000')
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--bs', type=int, default=32)
	parser.add_argument('--lr', type=float, default=0.05)
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('policy', type=str, default='boosted')

	spec_policy = sys.argv[1]
	policy_module = importlib.import_module(f'trainer.policy.{spec_policy}')
	parser = policy_module.add_args(parser)

	# limit PyTorch thread usage to avoid excessive CPU spin
	max_threads = min(4, max(1, (os.cpu_count() or 1)))
	torch.set_num_threads(max_threads)
	os.environ.setdefault('OMP_NUM_THREADS', str(max_threads))
	os.environ.setdefault('MKL_NUM_THREADS', str(max_threads))
	
	args = parser.parse_args()
	args.scales = [1, 0.3, 0.5, 0.7, 0.9]
	args.device = torch.device('cuda:' + args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')
	
	if CIFAR100 in args.dataset or SVHN in args.dataset or SPEECHCMDS in args.dataset:
		args.train_prefix = f'./dataset/{args.dataset}/train/'
	elif GLUE in args.dataset:
		args.train_prefix = f'./dataset/glue/{args.dataset}/train/'

	teachers_dir = args.teachers_dir
	# 获取teachers_dir下面所有文件的名称
	file_names = os.listdir(teachers_dir)
	# 筛掉文件夹
	file_names = [f for f in file_names if os.path.isfile(os.path.join(teachers_dir, f))]
	# print('Found teacher files:', file_names)
	model_names = list(set(['.'.join(f.split('.')[:-1]) for f in file_names if 'eval' not in f and '.' in f and '.png' not in f]))
	model_paths = [f'./{teachers_dir}/{model_name}' for model_name in model_names]
	for teacher_path in model_paths:
		if 'G' in teacher_path or 'loss' in teacher_path or 'acc' in teacher_path or 'distance' in teacher_path or 'budget' in teacher_path:
			continue
		# TODO
		if 'reefl' in teacher_path:
			continue
		if 'depthfl' not in teacher_path and 'darkfl' not in teacher_path:
			continue
		print('Processing teacher model:', teacher_path)
		args.teacher_pth = teacher_path + '.pth'
		args.teacher_config = teacher_path + '.json'
		args.output_prefix = teacher_path + '_variants/variant'
		teacher_distillation(args)


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
