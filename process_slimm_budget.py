import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import List, Dict, Any

# 模仿 gui/evaluate.py 的 setup_ax 风格
def setup_ax(ax):
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei'] # Use SimHei for Chinese
    plt.rcParams['axes.unicode_minus'] = False
    
    RATIO = 1.7
    TEXT_SIZE = 40/RATIO
    TRICKS_SIZE = 22/RATIO

    ax.set_facecolor("#EAEAF2")
    ax.grid(color="white", linestyle="-", linewidth=1.0, zorder=0)
    ax.tick_params(axis="x", which="both", top=False, bottom=False, length=0)
    ax.tick_params(axis="y", which="both", left=False, right=False, length=0)
    plt.tick_params(axis='both', which='major', labelsize=TRICKS_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Average Budget (in MUL-ADD) $\\times 10^{10}$", fontsize=TEXT_SIZE)
    ax.set_ylabel("Accuracy (%)", fontsize=TEXT_SIZE)

# 核心绘图函数，完全模仿 gui/evaluate.py 的 _plot_budget_curves 逻辑
def _plot_budget_curves_mock(
    curve_data_list: List[Dict],
    fitting_color: str = "#FF4500", # OrangeRed - 极高辨识度的橙红色
    fill_alpha: float = 0.1,
    title: str = ""
):
    RATIO = 1.7
    TEXT_SIZE = 40/RATIO
    TRICKS_SIZE = 22/RATIO

    fig, ax = plt.subplots(figsize=(8, 5.5)) # 稍微压缩一点高度
    setup_ax(ax)

    fitting_curves_y = []
    all_x = []
    for c in curve_data_list:
        all_x.extend(c["x"].tolist())
    
    if not all_x:
        return fig, ax
        
    x_grid = np.linspace(min(all_x), max(all_x), 500)

    for data in curve_data_list:
        x, y = data["x"], data["y"]
        color = data.get("color")
        marker = data.get("marker", "o")
        style = data.get("style", "-")
        label = data.get("label", "")
        is_fitting = data.get("is_fitting", False)
        linewidth = data.get("linewidth", 1.5)
        alpha = data.get("alpha", 1.0) 

        if not is_fitting:
            line, = ax.plot(
                x, y,
                color="#1f77b4", # DarkSlateGray - 与红色/彩色形成鲜明对比的深灰色
                marker=marker, linestyle=style,
                markeredgecolor="white", markeredgewidth=1,
                markersize=7,
                linewidth=3.0, # 原始曲线显著加粗
                label=label,
                alpha=alpha,
                zorder=3
            )
            # 原始曲线下方使用浅蓝色/灰色填充，区分开红色
            ax.fill_between(x, y, 0, color="#1f77b4", alpha=0.25, zorder=1)
        else:
            # 改进拟合：为了让红线(Maximum Fitted)能延伸到最右侧，右侧使用最后一个点进行填充(水平外推)
            # 而左侧不填充(NaN)，避免干扰左侧起点的包络线计算
            f_interp = interp1d(x, y, bounds_error=False, fill_value=(np.nan, y[-1]))
            fitting_curves_y.append(f_interp(x_grid))
            ax.plot(
                x, y, 
                color=color, marker=marker, linestyle=":", 
                markeredgecolor="white", markeredgewidth=0.8,
                markersize=6,
                linewidth=1.2,
                alpha=0.7, 
                label=label,
                zorder=2,
            )

    if fitting_curves_y:
        # 只在有数据的区域进行 max 计算
        fitting_curves_y_arr = np.array(fitting_curves_y)
        # 将 NaN 值保留，nanmax 会忽略它们，只有当所有候选曲线在某处都是 NaN 时，结果才是 NaN
        max_y = np.nanmax(fitting_curves_y_arr, axis=0)
        
        # 找到有效数据的索引范围
        valid_idx = np.where(~np.isnan(max_y))[0]
        if len(valid_idx) > 0:
            start, end = valid_idx[0], valid_idx[-1]
            ax.plot(x_grid[start:end+1], max_y[start:end+1], color=fitting_color, linewidth=3, label="Maximum Fitted", zorder=4, linestyle="--")
            # 显著加深拟合曲线下方的填充颜色，使用暖色调
            ax.fill_between(x_grid[start:end+1], max_y[start:end+1], 0, color="#FF6347", alpha=0.25, zorder=1)
        
        all_y = [v for c in curve_data_list for v in c["y"].tolist()]
        # 排除 nan 后的 max_y
        valid_max_y = max_y[~np.isnan(max_y)]
        if len(valid_max_y) > 0:
            all_y.extend(valid_max_y.tolist())
        ax.set_ylim(min(all_y) * 0.98, max(all_y) * 1.02) # 更加紧致的 y 轴范围

    if title:
        ax.set_title(title, fontsize=TEXT_SIZE)
        
    ax.legend(frameon=False, fontsize=TRICKS_SIZE - 2, loc='lower right', ncol=1)
        
    fig.set_tight_layout(True)
    return fig, ax

def process_and_plot(target_root, origin_root, output_folder, datasets):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_exits_configs = ["[2, 5, 8, 11]", "[2, 5, 8]", "[2, 5]"]
    
    # 颜色配置直接搬过来
    db_color = {
        "darkflpa2": "red",
        "darkflpg": "red",
        "reefl": "blue",
        "exclusivefl": "green",
        "scalefl": "orange"
    }

    for ds in datasets:
        ds_output = os.path.join(output_folder, f"plots_{ds}")
        if not os.path.exists(ds_output):
            os.makedirs(ds_output)
            
        target_dir = os.path.join(target_root, "full_boosted", ds)
        origin_dir = os.path.join(origin_root, "full_boosted", ds)

        for exits_config in target_exits_configs:
            slimm_groups = {}
            origin_results = {}
            
            if os.path.exists(target_dir):
                # 预先列出文件，避免在循环中重复列出
                files = os.listdir(target_dir)
                # 支持 [0.5-0.7-1.0] 等格式，需要包括连字符-
                pattern = re.compile(rf'^.*_slim_\[(?P<slim_config>[\d\.,\s\-]+)\]_slim_(?P<ratio>[\d\.]+)_exits_{re.escape(exits_config)}_eval\.json$')
                for fname in files:
                    match = pattern.match(fname)
                    if match:
                        alg_name = fname.split('_')[0]
                        if "darkfl" not in alg_name: continue
                        
                        file_path = os.path.join(target_dir, fname)
                        try:
                            with open(file_path, 'r') as f: data = json.load(f)
                            x, y = np.asarray(data["flops"])/1e9, np.asarray(data["test"])
                            
                            # 归一化 slim_config 到排序后的连字符形式
                            try:
                                inner = match.group('slim_config').replace(' ', '')
                                parts = re.split(r'[,\-]', inner)
                                parts = sorted([p for p in parts if p], key=float)
                                slim_ratios_config = f"[{'-'.join(parts)}]"
                            except Exception:
                                slim_ratios_config = f"[{match.group('slim_config')}]"
                        
                            need = ["[0.8-0.9-1.0]","[0.8-1.0]","[0.85-1.0]","[0.9-1.0]","[0.9-0.95-1.0]"]
                            if slim_ratios_config not in need:
                                continue
                            
                            key = (alg_name, slim_ratios_config)
                            if key not in slimm_groups: slimm_groups[key] = []
                            slimm_groups[key].append({
                                "x": x, "y": y, "ratio": float(match.group('ratio')), "alg": alg_name
                            })
                        except: pass

            if os.path.exists(origin_dir):
                files = os.listdir(origin_dir)
                pattern = re.compile(rf'^(?P<alg>[^_]+)_.*_exits_{re.escape(exits_config)}_eval\.json$')
                for fname in files:
                    match = pattern.match(fname)
                    if match:
                        alg = match.group('alg')
                        if "darkfl" not in alg: continue
                        try:
                            with open(os.path.join(origin_dir, fname), 'r') as f: data = json.load(f)
                            origin_results[alg] = {
                                "x": np.asarray(data["flops"])/1e9, "y": np.asarray(data["test"]), "alg": alg
                            }
                        except: pass

            for (alg, slim_cfg), curves in slimm_groups.items():
                curve_inputs = []
                orig = origin_results.get(alg)
                orig_color = "#1f77b4"
                if orig:
                    alg_label = "DarkDistill-PL" if alg == "darkflpa2" else "DarkDistill" if alg == "darkflpg" else alg
                    curve_inputs.append({
                        "x": orig["x"], "y": orig["y"], 
                        "color": orig_color, "label": f"{alg_label} (Original)",
                        "is_fitting": False, "linewidth": 2, "marker": "X",
                        "style": "-"
                    })
                
                curves = sorted(curves, key=lambda x: x["ratio"])
                
                # 既然每张图里只有一种算法的不同 ratio，我们可以直接使用区分度高的独立颜色
                colors_pool = ["#17becf", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
                marker_pool = ["o", "s", "^", "D", "v", "p", "*", "X"]
                
                for i, c in enumerate(curves):
                    # 为不同的 ratio 分配完全不同的颜色(从 tab10 色板取)和不同的形状
                    final_color = colors_pool[i % len(colors_pool)]
                    mrk = marker_pool[i % len(marker_pool)]
                        
                    curve_inputs.append({
                        "x": c["x"], "y": c["y"], 
                        "color": final_color, "marker": mrk,
                        "label": f"Slim ratio {c['ratio']}", 
                        "is_fitting": True,
                        "alpha": 0.8,         # 稍微提高点透明度避免喧宾夺主
                        "linewidth": 1.5,
                        "style": "--"         # 更改为虚线
                    })

                fig, ax = _plot_budget_curves_mock(
                    curve_inputs, 
                    fitting_color=db_color.get(alg, "red"),
                )
                
                c_slug = exits_config.replace('[','').replace(']','').replace(' ','').replace(',','_')
                s_slug = slim_cfg.replace('[','').replace(']','').replace('-','_').replace(',','_')
                path_base = os.path.join(ds_output, f"plot_{alg}_slim{s_slug}_exits{c_slug}")
                
                print(f"Saving to {path_base}.pdf")
                fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
                plt.close(fig)

if __name__ == "__main__":
    process_and_plot("EXPS2_inout/BASE_CIFAR", "EXPS2_inout/BASE_CIFAR_ORIGIN", 
                     "eval_output/CIFAR_Exits_Plots_MockGUI", ["noniid1000", "noniid1", "noniid0.1"])
