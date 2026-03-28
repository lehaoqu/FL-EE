import os
import re
import json
import copy
import math
import random
import argparse
from itertools import product as itertools_product
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from eval import Eval, Tester
from utils.options import args_parser
from utils.modelload.modelloader import load_model_eval
from utils.modelload.slimmable import set_width_ratio, SlimmableLinear, SlimmableConv2d
from utils.train_utils import area_under_fitted_curve, get_flops


def _seed_all(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def _layer_to_block(layer_idx: int, exits: List[int]) -> int:
	for block_idx, exit_layer in enumerate(exits):
		if layer_idx <= exit_layer:
			return block_idx
	return len(exits) - 1


def _module_block_index(name: str, exits: List[int]) -> int:
	m = re.search(r"vit\.encoder\.layer\.(\d+)", name)
	if m:
		return _layer_to_block(int(m.group(1)), exits)
	if "vit.embeddings" in name:
		return 0
	return len(exits) - 1


def _effective_module_params(module: nn.Module, ratio: float) -> int:
	if isinstance(module, SlimmableLinear):
		in_features = module.in_features
		out_features = module.out_features

		if module.fix_in_dim:
			in_dim = in_features
		else:
			in_dim = int(in_features * ratio)

		if module.fix_out_dim:
			out_dim = out_features
		else:
			out_dim = int(out_features * ratio)

		in_dim = max(1, min(in_dim, in_features))
		out_dim = max(1, min(out_dim, out_features))
		return out_dim * in_dim + (out_dim if module.bias is not None else 0)

	if isinstance(module, SlimmableConv2d):
		in_channels = module.in_channels
		out_channels = module.out_channels
		cout = max(1, min(int(out_channels * ratio), out_channels))
		return cout * in_channels * module.kernel_size[0] * module.kernel_size[1] + (cout if module.bias is not None else 0)

	if isinstance(module, nn.Linear):
		return module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)

	if isinstance(module, nn.Conv2d):
		return module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)

	if isinstance(module, nn.LayerNorm):
		return module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)

	return 0


def estimate_effective_params(model: nn.Module, block_ratios: List[float], exits: List[int]) -> int:
	total = 0
	for name, module in model.named_modules():
		block_idx = _module_block_index(name, exits)
		ratio = block_ratios[block_idx]
		total += _effective_module_params(module, ratio)
	return int(total)


def calc_logits_blockwise(
	model: nn.Module,
	tester: Tester,
	dataloader,
	block_ratios: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
	model.eval()
	n_exits = len(model.config.exits)
	all_sample_exits_logits = [[] for _ in range(n_exits)]
	all_sample_targets = []

	with torch.no_grad():
		for data in dataloader:
			batch, y = tester.adapt_batch(data)
			all_sample_targets.append(y)

			full_embeddings = model(**batch, rt_embedding=True)
			prev_feature = None

			for block_idx in range(n_exits):
				ratio = block_ratios[block_idx]
				set_width_ratio(ratio, model)

				if block_idx == 0:
					exits_logits, _exits_feature, exits_full_feature = model(
						**batch,
						input_block=block_idx,
						stop_exit=block_idx,
					)
				else:
					latent_batch = {'pixel_values': prev_feature}
					exits_logits, _exits_feature, exits_full_feature = model(
						**latent_batch,
						input_block=block_idx,
						stop_exit=block_idx,
						is_latent=True,
					)

				all_sample_exits_logits[block_idx].append(exits_logits[0])
				prev_feature = exits_full_feature[0].detach()

			set_width_ratio(1.0, model)

	for i in range(n_exits):
		all_sample_exits_logits[i] = torch.cat(all_sample_exits_logits[i], dim=0)

	size = (
		len(all_sample_exits_logits),
		all_sample_exits_logits[0].size(0),
		all_sample_exits_logits[0].size(1),
	)
	preds = torch.zeros(size=size, device=tester.device)
	for i in range(len(all_sample_exits_logits)):
		preds[i] = all_sample_exits_logits[i]

	targets = torch.cat(all_sample_targets, dim=0)
	return preds, targets


def _get_exit_classifier_flops(model: nn.Module, device, exit_idx: int) -> float:
	"""Profile FLOPs of a single exit head (LayerNorm + Linear) for one sample."""
	from thop import profile

	layer_idx = model.config.exits[exit_idx]
	exit_layer = model.vit.encoder.layer[layer_idx]

	class _ExitHeadWrapper(nn.Module):
		def __init__(self, ln: nn.Module, fc: nn.Module):
			super().__init__()
			self.ln = ln
			self.fc = fc

		def forward(self, x):
			return self.fc(self.ln(x))

	head = _ExitHeadWrapper(exit_layer.classifier_layernorm, exit_layer.classifier).to(device)
	dummy_x = torch.zeros(1, model.config.hidden_size, device=device)
	macs, _ = profile(head, inputs=(dummy_x,), verbose=False)
	return float(macs * 2)


def _compute_exit_flops(
	model: nn.Module,
	tester: Tester,
	block_ratios: List[float],
) -> List[float]:
	"""Compute cumulative per-exit FLOPs for a given block_ratios config."""
	cls_flops = [_get_exit_classifier_flops(model, tester.device, i) for i in range(tester.n_exits)]
	flops = []
	cum_backbone_flops = 0.0
	for i in range(tester.n_exits):
		set_width_ratio(block_ratios[i], model)
		if i == 0:
			block_total = get_flops(tester.args, model, stop_exit=i)
		else:
			block_total = get_flops(tester.args, model, stop_exit=i, input_feature=True)
		block_backbone = max(0.0, float(block_total) - float(cls_flops[i]))
		cum_backbone_flops += block_backbone
		flops.append(cum_backbone_flops + float(cls_flops[i]))
	set_width_ratio(1.0, model)
	return flops


def eval_block_config(
	model: nn.Module,
	tester: Tester,
	valid_loader,
	test_loader,
	block_ratios: List[float],
	flops_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
	valid_exits_preds, valid_targets = calc_logits_blockwise(model, tester, valid_loader, block_ratios)
	test_exits_preds, test_targets = calc_logits_blockwise(model, tester, test_loader, block_ratios)

	flops = _compute_exit_flops(model, tester, block_ratios)

	rnd = 40
	acc_list = []
	exp_flops_list = []

	for p in range(1, rnd):
		_p = torch.tensor([p * (1.0 / (rnd / 2))], dtype=torch.float32).to(tester.device)
		probs = torch.exp(torch.log(_p) * torch.tensor([(i + 1) * 4 for i in range(tester.n_exits)]).to(tester.device))
		probs /= probs.sum()

		_acc_val, _exp_val, T = tester.dynamic_eval_find_threshold(valid_exits_preds, valid_targets, probs, flops)
		acc_test, exp_flops = tester.dynamic_eval_with_threshold(test_exits_preds, test_targets, flops, T)
		acc_list.append(float(acc_test))
		exp_flops_list.append(float(exp_flops))

	if flops_bounds is not None:
		min_f, max_f = flops_bounds
		x_arr = np.asarray(exp_flops_list, dtype=float)
		y_arr = np.asarray(acc_list, dtype=float)
		order = np.argsort(x_arr)
		x_arr = x_arr[order]
		y_arr = y_arr[order]
		x_grid = np.linspace(min_f, max_f, num=max(200, len(acc_list) * 5))
		# left=0: pad with 0 for flops below this config's range
		# right=last acc: extend with last acc for flops above this config's range
		y_interp = np.interp(x_grid, x_arr, y_arr, left=0.0, right=float(y_arr[-1]))
		auc, avg_acc = area_under_fitted_curve(y_interp.tolist(), x_grid.tolist())
	else:
		auc, avg_acc = area_under_fitted_curve(acc_list, exp_flops_list)

	params = estimate_effective_params(model, block_ratios, list(model.config.exits))
	return {
		'budgeted_acc': float(avg_acc),
		'budgeted_auc': float(auc),
		'params': float(params),
		'flops': exp_flops_list,
		'acc_curve': acc_list,
	}


def calc_block_delta_dict(
	model: nn.Module,
	tester: Tester,
	valid_loader,
	test_loader,
	slim_ratios: List[float],
	flops_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[int, List[Dict[str, float]]], Dict[Tuple[float, ...], Dict[str, float]]]:
	n_blocks = len(model.config.exits)
	min_ratio = min(slim_ratios)
	eval_cache: Dict[Tuple[float, ...], Dict[str, float]] = {}

	def _get_eval(cfg: List[float]) -> Dict[str, float]:
		print(f"Evaluating config: {cfg}")
		key = tuple(cfg)
		if key not in eval_cache:
			eval_cache[key] = eval_block_config(model, tester, valid_loader, test_loader, cfg, flops_bounds=flops_bounds)
		print(eval_cache[key])
		return eval_cache[key]

	base_cfg = [min_ratio] * n_blocks
	_get_eval(base_cfg)

	block_delta_dict: Dict[int, List[Dict[str, float]]] = {}
	sorted_ratios = sorted(slim_ratios)

	for block_idx in range(n_blocks):
		steps = []
		prev_ratio = sorted_ratios[0]
		prev_cfg = [min_ratio] * n_blocks
		prev_cfg[block_idx] = prev_ratio
		prev_eval = _get_eval(prev_cfg)

		for ratio in sorted_ratios[1:]:
			cur_cfg = [min_ratio] * n_blocks
			cur_cfg[block_idx] = ratio
			cur_eval = _get_eval(cur_cfg)

			delta_auc = cur_eval['budgeted_auc'] - prev_eval['budgeted_auc']
			delta_params = cur_eval['params'] - prev_eval['params']
			score = (delta_auc / delta_params) if delta_params > 0 else -1e9

			steps.append({
				'from_ratio': float(prev_ratio),
				'to_ratio': float(ratio),
				'delta_auc': float(delta_auc),
				'delta_params': float(delta_params),
				'gain_per_param': float(score),
			})

			prev_ratio = ratio
			prev_eval = cur_eval

		block_delta_dict[block_idx] = steps

	return block_delta_dict, eval_cache


def greedy_upgrade_for_budget(
	block_delta_dict: Dict[int, List[Dict[str, float]]],
	min_ratio_cfg: List[float],
	budget_params: float,
) -> List[float]:
	n_blocks = len(min_ratio_cfg)
	cfg = list(min_ratio_cfg)
	step_ptr = [0] * n_blocks

	consumed = 0.0
	while True:
		candidates = []
		for block_idx in range(n_blocks):
			ptr = step_ptr[block_idx]
			steps = block_delta_dict[block_idx]
			if ptr >= len(steps):
				continue
			step = steps[ptr]
			candidates.append((step['gain_per_param'], block_idx, step))

		if not candidates:
			break

		candidates.sort(key=lambda t: t[0], reverse=True)
		best_score, best_block, best_step = candidates[0]

		next_consumed = consumed + best_step['delta_params']
		if next_consumed > budget_params:
			break

		consumed = next_consumed
		cfg[best_block] = best_step['to_ratio']
		step_ptr[best_block] += 1

	return cfg


def build_budget_curve(
	model: nn.Module,
	tester: Tester,
	valid_loader,
	test_loader,
	slim_ratios: List[float],
	block_delta_dict: Dict[int, List[Dict[str, float]]],
	eval_cache: Dict[Tuple[float, ...], Dict[str, float]],
	num_points: int = 50,
	flops_bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, List[float]]:
	n_blocks = len(model.config.exits)
	min_ratio = min(slim_ratios)
	max_ratio = max(slim_ratios)
	min_cfg = [min_ratio] * n_blocks
	max_cfg = [max_ratio] * n_blocks

	min_key = tuple(min_cfg)
	max_key = tuple(max_cfg)
	if min_key not in eval_cache:
		eval_cache[min_key] = eval_block_config(model, tester, valid_loader, test_loader, min_cfg, flops_bounds=flops_bounds)
	if max_key not in eval_cache:
		eval_cache[max_key] = eval_block_config(model, tester, valid_loader, test_loader, max_cfg, flops_bounds=flops_bounds)

	min_params = eval_cache[min_key]['params']
	max_params = eval_cache[max_key]['params']

	budgets = np.linspace(min_params, max_params, num=max(2, num_points)).tolist()
	curve_params = []
	curve_auc = []
	curve_acc = []
	curve_cfg = []

	for budget in budgets:
		extra_budget = max(0.0, budget - min_params)
		cfg = greedy_upgrade_for_budget(block_delta_dict, min_cfg, extra_budget)
		key = tuple(cfg)
		if key not in eval_cache:
			eval_cache[key] = eval_block_config(model, tester, valid_loader, test_loader, cfg, flops_bounds=flops_bounds)
		metric = eval_cache[key]

		curve_params.append(float(budget))
		curve_auc.append(float(metric['budgeted_auc']))
		curve_acc.append(float(metric['budgeted_acc']))
		curve_cfg.append([float(x) for x in cfg])

	curve_auc_area, _curve_avg = area_under_fitted_curve(curve_auc, curve_params)
	return {
		'params': curve_params,
		'auc': curve_auc,
		'acc': curve_acc,
		'cfg': curve_cfg,
		'curve_auc': float(curve_auc_area),
		'min_params': float(min_params),
		'max_params': float(max_params),
	}


_EXPS2_INOUT_BASE = "/home/qvlehao/FL-EE/EXPS2_inout/BASE_CIFAR/full_boosted"


def _find_noniid_dir(model_path_prefix: str) -> str:
	"""Extract noniid directory name (e.g. 'noniid1000') from model path or basename."""
	parts = model_path_prefix.replace('\\', '/').split('/')
	for part in reversed(parts):
		if re.match(r'noniid[\d.]+$', part):
			return part
	m = re.search(r'(noniid[\d.]+)', os.path.basename(model_path_prefix))
	if m:
		return m.group(1)
	return 'noniid1000'


def _load_uniform_baseline(
	model: nn.Module,
	tester: Tester,
	model_path_prefix: str,
	slim_ratios: List[float],
	flops_bounds: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
	"""Load per-ratio uniform eval jsons from EXPS2_inout, recompute AUC with global
	flops_bounds, and estimate params for each uniform-ratio config.

	Returns list of dicts sorted by params:
	  {'ratio', 'params', 'auc', 'avg_acc'}
	"""
	basename = os.path.basename(model_path_prefix)
	noniid_dir = _find_noniid_dir(model_path_prefix)
	n_blocks = len(model.config.exits)
	exits_str = str(list(model.config.exits))  # e.g. '[2, 5, 8]'
	exps2_dir = os.path.join(_EXPS2_INOUT_BASE, noniid_dir)

	results = []
	for ratio in sorted(slim_ratios):
		ratio_str = str(float(ratio))
		eval_json = os.path.join(exps2_dir, f"{basename}_slim_{ratio_str}_exits_{exits_str}_eval.json")
		if not os.path.exists(eval_json):
			print(f"[WARN] Uniform baseline not found: {eval_json}")
			continue
		with open(eval_json, 'r') as f:
			dct = json.load(f)

		acc_list = dct['test']
		flops_list = dct['flops']
		# print(acc_list)
		# print(flops_list)

		if flops_bounds is not None:
			min_f, max_f = flops_bounds
			x_arr = np.asarray(flops_list, dtype=float)
			y_arr = np.asarray(acc_list, dtype=float)
			order = np.argsort(x_arr)
			x_arr, y_arr = x_arr[order], y_arr[order]
			x_grid = np.linspace(min_f, max_f, num=max(200, len(acc_list) * 5))
			y_interp = np.interp(x_grid, x_arr, y_arr, left=0.0, right=float(y_arr[-1]))
			auc, avg_acc = area_under_fitted_curve(y_interp.tolist(), x_grid.tolist())
		else:
			auc, avg_acc = area_under_fitted_curve(acc_list, flops_list)
		print(f"Uniform ratio={ratio}: AUC={auc:.6f}, avg_acc={avg_acc:.4f}")
		uniform_cfg = [float(ratio)] * n_blocks
		params = estimate_effective_params(model, uniform_cfg, list(model.config.exits))
		results.append({'ratio': float(ratio), 'params': float(params), 'auc': float(auc), 'avg_acc': float(avg_acc)})

	return sorted(results, key=lambda r: r['params'])


def _step_auc(
	uniform_results: List[Dict],
	x_min: float,
	x_max: float,
) -> Tuple[float, float]:
	"""Compute (total_area, avg) of the step-function baseline over [x_min, x_max].

	Step rule: for budget in [params(r_i), params(r_{i+1})), achievable auc = auc(r_i).
	Below the smallest ratio's params the value is 0.
	"""
	sorted_r = sorted(uniform_results, key=lambda r: r['params'])
	prev_x = x_min
	prev_y = 0.0
	total = 0.0
	for r in sorted_r:
		p, a = r['params'], r['auc']
		right = min(p, x_max)
		if right > prev_x:
			total += prev_y * (right - prev_x)
		if p >= x_max:
			prev_x = x_max
			break
		prev_x = p
		prev_y = a
	# tail: from last ratio's params to x_max
	if prev_x < x_max:
		total += prev_y * (x_max - prev_x)
	width = x_max - x_min
	avg = total / width if width > 0 else 0.0
	return total, avg


def _plot_comparison_curve(
	curve: Dict,
	uniform_results: List[Dict],
	out_path: str,
	model_name: str,
) -> Dict[str, float]:
	"""Plot blockwise params-AUC curve vs uniform-ratio step baseline. Return both AUCs."""
	block_params = curve['params']
	block_auc_vals = curve['auc']
	x_min = float(min(block_params))
	x_max = float(max(block_params))

	blockwise_curve_auc = float(curve['curve_auc'])

	# Build step-function xs/ys for plotting
	sorted_r = sorted(uniform_results, key=lambda r: r['params'])
	step_xs = [x_min] + [r['params'] for r in sorted_r] + [x_max]
	step_ys = [0.0] + [r['auc'] for r in sorted_r] + \
	          [sorted_r[-1]['auc'] if sorted_r else 0.0]

	uniform_area, _ = _step_auc(uniform_results, x_min, x_max)

	os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(block_params, block_auc_vals, marker='o', markersize=4, linewidth=1.5,
	        label=f'Blockwise (curve AUC={blockwise_curve_auc:.3e})')
	ax.step(step_xs, step_ys, where='post', linewidth=1.5, linestyle='--',
	        label=f'Uniform ratio (curve AUC={uniform_area:.3e})')
	for r in sorted_r:
		ax.annotate(f"r={r['ratio']}", xy=(r['params'], r['auc']),
		            xytext=(5, 5), textcoords='offset points', fontsize=8)

	ax.set_xlabel('Params')
	ax.set_ylabel('Budgeted AUC')
	ax.set_title(f'Blockwise vs Uniform Ratio\n{model_name}')
	ax.legend(fontsize=9)
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_path, dpi=200)
	plt.close(fig)

	return {'blockwise_curve_auc': blockwise_curve_auc, 'uniform_curve_auc': uniform_area}


def _plot_budget_curve(params: List[float], auc: List[float], out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.figure(figsize=(7, 5))
	plt.plot(params, auc, marker='o', linewidth=1.5)
	plt.xlabel('Params')
	plt.ylabel('Best Budgeted AUC')
	plt.title('Greedy Blockwise Params-AUC Curve')
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()


def run_blockwise_eval(eval_obj: Eval, model_path_prefix: str, budget_points: int = 50) -> None:
	model = load_model_eval(eval_obj.args, model_path=model_path_prefix + '.pth', config_path=model_path_prefix + '.json')
	model.to(eval_obj.device)
	model.eval()

	if not getattr(model.config, 'slimmable', False):
		eval_obj._log(f"[SKIP] Not slimmable: {os.path.basename(model_path_prefix)}")
		return

	slim_ratios = sorted([float(r) for r in model.config.slim_ratios])
	n_blocks = len(model.config.exits)
	eval_obj.args.n_exits = n_blocks
	tester = Tester(model, eval_obj.args, measure_flops=False)

	eval_obj._log(f"[INFO] model={os.path.basename(model_path_prefix)}, blocks={n_blocks}, slim_ratios={slim_ratios}")

	# # Compute global flops bounds from the actual exp_flops_list of min/max configs.
	# # Run inference on extreme configs (no bounds yet) to get the real expected-flops range.
	min_cfg_list = [min(slim_ratios)] * n_blocks
	max_cfg_list = [max(slim_ratios)] * n_blocks
	_min_metric = eval_block_config(model, tester, eval_obj.valid_dataloader, eval_obj.test_dataloader, min_cfg_list)
	_max_metric = eval_block_config(model, tester, eval_obj.valid_dataloader, eval_obj.test_dataloader, max_cfg_list)
	global_min_flops = float(min(_min_metric['flops']))
	global_max_flops = float(max(_max_metric['flops']))
	flops_bounds = (global_min_flops, global_max_flops)
	print(f'bounds: {flops_bounds}')
	eval_obj._log(f"[INFO] global flops bounds (from exp_flops_list): [{global_min_flops:.2f}, {global_max_flops:.2f}]")

	# block_delta_dict, eval_cache = calc_block_delta_dict(
	# 	model,
	# 	tester,
	# 	eval_obj.valid_dataloader,
	# 	eval_obj.test_dataloader,
	# 	slim_ratios,
	# 	flops_bounds=flops_bounds,
	# )

	# min_cfg = [min(slim_ratios)] * n_blocks
	# min_metric = eval_cache[tuple(min_cfg)]

	# curve = build_budget_curve(
	# 	model,
	# 	tester,
	# 	eval_obj.valid_dataloader,
	# 	eval_obj.test_dataloader,
	# 	slim_ratios,
	# 	block_delta_dict,
	# 	eval_cache,
	# 	num_points=budget_points,
	# 	flops_bounds=flops_bounds,
	# )

	# # --- Print delta_auc table (per block, per upgrade step) ---
	# eval_obj._log("\n" + "=" * 60)
	# eval_obj._log("Block delta_auc per upgrade step:")
	# header = f"{'Block':>6} | {'Step':>20} | {'delta_auc':>12} | {'delta_params':>14} | {'gain/param':>12}"
	# eval_obj._log(header)
	# eval_obj._log("-" * len(header))
	# for block_idx, steps in block_delta_dict.items():
	# 	for step in steps:
	# 		eval_obj._log(
	# 			f"{block_idx:>6} | {step['from_ratio']:>8.2f}->{step['to_ratio']:>8.2f} | "
	# 			f"{step['delta_auc']:>12.4f} | {step['delta_params']:>14.0f} | {step['gain_per_param']:>12.4e}"
	# 		)

	# # --- Print greedy config evolution as budget increases ---
	# eval_obj._log("\n" + "=" * 60)
	# eval_obj._log("Greedy config evolution along budget curve:")
	# prev_cfg = None
	# for budget, cfg in zip(curve['params'], curve['cfg']):
	# 	if cfg != prev_cfg:
	# 		eval_obj._log(f"  budget={budget:.0f}  cfg={cfg}")
	# 		prev_cfg = cfg
	# eval_obj._log("=" * 60 + "\n")

	# out_dir = eval_obj.eval_dir
	# base_name = os.path.basename(model_path_prefix)
	# out_json = os.path.join(out_dir, f"{base_name}_slim_block_budgeted.json")
	# out_png = os.path.join(out_dir, f"{base_name}_slim_block_budgeted.png")

	# result = {
	# 	'model': base_name,
	# 	'exits': [int(e) for e in model.config.exits],
	# 	'slim_ratios': slim_ratios,
	# 	'base': {
	# 		'cfg': min_cfg,
	# 		'budgeted_acc': float(min_metric['budgeted_acc']),
	# 		'budgeted_auc': float(min_metric['budgeted_auc']),
	# 		'params': float(min_metric['params']),
	# 	},
	# 	'block_delta_dict': block_delta_dict,
	# 	'budget_curve': curve,
	# }

	# Load uniform-ratio baseline from EXPS2_inout
	# Optionally set global FLOPs bounds for fair AUC comparison
	uniform_results = _load_uniform_baseline(
		model, tester, model_path_prefix, slim_ratios, flops_bounds=flops_bounds
	)

	# Comparison plot and AUC
	out_compare = os.path.join(out_dir, f"{base_name}_slim_block_vs_uniform.png")
	compare_metrics = {}
	if uniform_results:
		compare_metrics = _plot_comparison_curve(curve, uniform_results, out_compare, base_name)
		eval_obj._log(
			f"[COMPARE] blockwise_curve_auc={compare_metrics['blockwise_curve_auc']:.6f}, "
			f"uniform_curve_auc={compare_metrics['uniform_curve_auc']:.6f}"
		)
		result['uniform_baseline'] = uniform_results
		result['compare_auc'] = compare_metrics
	else:
		eval_obj._log("[COMPARE] No uniform baseline found, skipping comparison plot.")

	with open(out_json, 'w') as f:
		json.dump(result, f, indent=2)
	_plot_budget_curve(curve['params'], curve['auc'], out_png)

	eval_obj._log(f"[DONE] base_acc={min_metric['budgeted_acc']:.4f}, base_auc={min_metric['budgeted_auc']:.4f}, base_params={min_metric['params']:.0f}")
	eval_obj._log(f"[DONE] curve_auc={curve['curve_auc']:.6f}")
	eval_obj._log(f"[SAVE] {out_json}")
	eval_obj._log(f"[SAVE] {out_png}")
	if compare_metrics:
		eval_obj._log(f"[SAVE] {out_compare}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--budget_points', type=int, default=50)
    known_args, _ = parser.parse_known_args()

    args = args_parser()
    
    # --- Default Model Override ---
    default_model = "/home/qvlehao/FL-EE/EXPS3_teacher_train_embedding_slim_extra/TEST_L_1_F_3/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_slim_[0.8-0.9-1.0].pth"
    if not os.path.exists(args.load_path):
        args.load_path = default_model
    # ------------------------------

    _seed_all(args.seed)
    args.img_dir = args.suffix + '/img'
    eval_obj = Eval(args=args)

    if os.path.isfile(args.load_path):
        model_paths = ['.'.join(args.load_path.split('.')[:-1])]
    else:
        file_names = os.listdir(args.suffix)
        model_names = list(set([
            '.'.join(f.split('.')[:-1])
            for f in file_names
            if 'eval' not in f and '.' in f and '.png' not in f and '.json' in f
        ]))
        model_paths = [f'./{args.suffix}/{name}' for name in model_names]

    for model_path in model_paths:
        if args.policy not in model_path and os.path.isdir(args.suffix):
            continue
        if ('G_' in model_path or 'loss' in model_path or 'acc' in model_path or 'distance' in model_path or 'budget' in model_path) and os.path.isdir(args.suffix):
            continue
        # if 'darkfl' not in model_path: continue
        eval_obj._log((f'eval slim-block model:{os.path.basename(model_path)}').center(120, '='))
        run_blockwise_eval(eval_obj, model_path, budget_points=known_args.budget_points)

