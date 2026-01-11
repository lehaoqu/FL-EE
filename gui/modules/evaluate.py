import streamlit as st
import os
import json
import subprocess
import selectors
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvalRunItem:
    label: str
    eval_json_path: str
    rel_dir: str
    policy: str
    dataset: str
    model: str
    ft: str
    mode: str


def _get_conda_python_path(env_name):
    """Get the Python executable path for a conda environment or system python"""
    if env_name == "System Python (python)":
        return "/usr/bin/python"
    
    try:
        result = subprocess.run(["conda", "info", "--envs", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout or "{}")
        envs = data.get("envs", [])
        
        for env_path in envs:
            if env_path.endswith(env_name) or env_path.endswith(f"/{env_name}"):
                python_path = os.path.join(env_path, "bin", "python")
                if os.path.exists(python_path):
                    return python_path
        return None
    except Exception:
        return None


def _scan_exps_suffixes():
    """Recursively scan fron t directory for available result suffixes"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exps_dir = os.path.join(project_root, "front-exps")
    
    suffixes = []
    
    if not os.path.exists(exps_dir):
        return suffixes
    
    # Walk through front-exps directory to find all subdirectories
    for root, dirs, files in os.walk(exps_dir):
        # Get relative path from front-exps
        rel_path = os.path.relpath(root, exps_dir)
        if rel_path != '.':
            # Check if this directory contains model files (.pth)
            has_models = any(f.endswith('.pth') for f in files)
            if has_models:
                suffixes.append(rel_path)
    
    return sorted(suffixes)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _front_exps_dir() -> str:
    return os.path.join(_project_root(), "front-exps")


def _infer_ft_from_rel_dir(rel_dir: str) -> str:
    parts = [p.lower() for p in rel_dir.split(os.sep) if p]
    # Common patterns: full_base/full_boosted, lora_base/lora_boosted, etc.
    if any("lora" in p for p in parts):
        return "lora"
    return "full"


def _parse_stem(stem: str) -> Tuple[str, str, str, str]:
    """Parse policy/dataset/model/mode from filename stem.

    Example stem:
      eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted
    """
    tokens = stem.split("_")
    policy = tokens[0] if tokens else ""

    model = ""
    model_idx: Optional[int] = None
    for i, tok in enumerate(tokens):
        if tok in {"vit", "bert"}:
            model = tok
            model_idx = i
            break

    dataset = ""
    if model_idx is not None and model_idx >= 2:
        dataset = "_".join(tokens[1:model_idx])
    elif len(tokens) >= 2:
        dataset = tokens[1]

    mode = ""
    for cand in ("boosted", "base", "l2w"):
        if cand in tokens:
            mode = cand
            break

    return policy, dataset, model, mode


def _is_eval_target_pth(filename: str) -> bool:
    if not filename.endswith(".pth"):
        return False
    # Generator checkpoints are not evaluated by eval.py in typical workflows.
    if "_G_" in filename:
        return False
    return True


def _scan_eval_complete_items() -> List[EvalRunItem]:
    """Scan front-exps and return eval-complete runs.

    Eval complete (per user spec): within a folder, every target .pth has a matching *_eval.json.
    """
    exps_dir = _front_exps_dir()
    if not os.path.exists(exps_dir):
        return []

    # Deduplicate by display label (path + algorithm) to avoid repeated-looking entries.
    items_by_label: Dict[str, EvalRunItem] = {}

    for root, _dirs, files in os.walk(exps_dir):
        pths = [f for f in files if _is_eval_target_pth(f)]
        if not pths:
            continue

        # A folder is considered eval-complete only if every target .pth has *_eval.json.
        stems = [os.path.splitext(f)[0] for f in pths]
        expected_eval = {f"{s}_eval.json" for s in stems}
        file_set = set(files)
        if not expected_eval.issubset(file_set):
            continue

        rel_dir = os.path.relpath(root, exps_dir)
        ft = _infer_ft_from_rel_dir(rel_dir)

        for stem in sorted(stems):
            eval_json = os.path.join(root, f"{stem}_eval.json")
            policy, dataset, model, mode = _parse_stem(stem)
            # User-facing label: path first, then algorithm.
            label = f"{rel_dir} | {policy}".strip()
            items_by_label.setdefault(
                label,
                EvalRunItem(
                    label=label,
                    eval_json_path=eval_json,
                    rel_dir=rel_dir,
                    policy=policy,
                    dataset=dataset,
                    model=model,
                    ft=ft,
                    mode=mode,
                ),
            )

    return sorted(items_by_label.values(), key=lambda x: x.label)


def _plot_budget_curves(selected: List[EvalRunItem]):
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import draw_budget as db
    except Exception:
        db = None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Match draw_budget aesthetics without saving/showing.
    ax.set_facecolor("#EAEAF2")
    ax.grid(color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(axis="x", which="both", top=False, bottom=False, length=0)
    ax.tick_params(axis="y", which="both", left=False, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for item in selected:
        try:
            with open(item.eval_json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            st.error(f"Failed to read: {item.eval_json_path} ({e})")
            continue

        flops = np.asarray(data.get("flops", []), dtype=float)
        acc = np.asarray(data.get("test", []), dtype=float)
        if flops.size == 0 or acc.size == 0:
            continue

        x = flops / 1e9  # GFLOPs
        color = (getattr(db, "COLOR", {}) if db else {}).get(item.policy, None)
        marker = (getattr(db, "MARKER", {}) if db else {}).get(item.policy, "o")
        style = (getattr(db, "STYLE", {}) if db else {}).get(item.policy, "-")
        name = (getattr(db, "NAMES", {}) if db else {}).get(item.policy, item.policy)
        label = f"{name} ({item.dataset}, {item.mode}, {item.ft})".strip()

        ax.plot(
            x,
            acc,
            color=color,
            marker=marker,
            linestyle=style,
            markeredgecolor="white",
            markeredgewidth=1,
            linewidth=2 if "darkfl" in item.policy else 1,
            label=label,
        )

    ax.set_xlabel("Budget (GFLOPs)")
    ax.set_ylabel("Accuracy (%)")

    # Put legend outside the plot to avoid covering curves.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=min(3, len(labels)),
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    st.pyplot(fig, use_container_width=True)


def show():
    """显示 Evaluate 页面"""
    st.header("评估")
    st.write("对 front-exps 目录下已训练模型运行评估")
    
    def _with_front_prefix(path: str) -> str:
        """Ensure suffix is under front-exps for downstream eval script."""
        return path if path.startswith("front-exps/") else f"front-exps/{path}"
    
    # --- Configuration ---
    with st.expander("评估配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模型与数据设置")
            
            # Scan available suffixes
            available_suffixes = _scan_exps_suffixes()
            
            if available_suffixes:
                st.caption(f"📁 在 front-exps 中发现 {len(available_suffixes)} 个已训练实验")
                suffix = st.selectbox(
                    "实验后缀",
                    available_suffixes,
                    help="从 front-exps 目录自动检测"
                )
            else:
                st.warning("⚠️ 在 front-exps/ 中未发现已训练模型。")
                suffix = st.text_input(
                    "实验后缀（手动输入）",
                    value="front-exps/test/cifar100/vit_base",
                    help="手动指定实验后缀（应以 front-exps/ 开头）"
                )
            
            # Scan available datasets from dataset directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dataset_dir = os.path.join(project_root, "dataset")
            available_datasets = []
            
            if os.path.exists(dataset_dir):
                for item in os.listdir(dataset_dir):
                    item_path = os.path.join(dataset_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                        available_datasets.append(item)
            
            if available_datasets:
                dataset = st.selectbox(
                    "数据集",
                    sorted(available_datasets),
                    help="用于评估的数据集"
                )
            else:
                dataset = st.text_input("数据集", value="cifar100_noniid1000")
            
            model = st.selectbox("模型", ["vit", "bert"])
        
        with col2:
            st.subheader("评估设置")
            st.caption("本界面固定使用算法 **eefl**；其他变体未列出。")
            algorithm = "eefl"
            boosted = st.selectbox("Boost 模式", ["boosted", "base", "l2w"])
            
            valid_ratio = st.number_input(
                "验证比例",
                value=0.2,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                help="用于验证的数据比例"
            )
            
            if_mode = st.selectbox(
                "推理模式",
                ["all", "anytime", "budgeted"],
                help="all: 同时包含 anytime 与 budgeted；anytime: 任意层提前退出；budgeted: 带预算约束"
            )
            
            device = st.text_input("设备 (GPU)", value="0", help="例如 0 表示 GPU:0，或填 cpu")
            
            fine_tuning = st.selectbox("微调方式", ["full", "lora"])
        
        st.divider()
        st.write("**生成的命令：**")
        suffix_prefixed = _with_front_prefix(suffix)
        cmd = (
            f"python -u eval.py {algorithm} {boosted} --suffix {suffix_prefixed} --device {device} "
            f"--dataset {dataset} --model {model} --valid_ratio {valid_ratio} --if_mode {if_mode} --ft {fine_tuning}"
        )
        st.code(cmd, language="bash")
    
    # --- Run Evaluation ---
    with st.expander("运行评估", expanded=True):
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 Using global conda environment: **{conda_env}**")
        else:
            st.caption("💡 可在“设置”页统一配置 Conda 环境")
        
        if st.button("运行评估", type="primary", use_container_width=True):
            st.info(f"使用 Conda 环境 '{conda_env}' 开始执行评估…")
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Replace python with the full conda python path
                direct_cmd = cmd.replace("python", python_path)
                st.info(f"Python 解释器：`{python_path}`")
                st.info(f"执行命令：`{direct_cmd}`")
                
                try:
                    # Force unbuffered Python output for real-time streaming
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    # Execute command with real-time output
                    # Use binary pipes + selectors to avoid blocking on readline()
                    process = subprocess.Popen(
                        direct_cmd,
                        shell=True,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                        env=env,
                    )
                    
                    # Display process ID
                    st.info(f"🔧 **进程号 (PID): {process.pid}** - 如需手动停止可执行：`kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []
                    
                    # Read output in real-time without nesting expanders
                    output_section = st.container()
                    output_section.subheader("📋 执行输出（实时）")
                    output_section.caption("显示评估进程的实时 stdout/stderr 输出。")
                    stdout_container = output_section.empty()
                    stderr_label_container = output_section.empty()
                    stderr_container = output_section.empty()

                    sel = selectors.DefaultSelector()
                    if process.stdout is not None:
                        sel.register(process.stdout, selectors.EVENT_READ, data="stdout")
                    if process.stderr is not None:
                        sel.register(process.stderr, selectors.EVENT_READ, data="stderr")

                    buffers = {"stdout": "", "stderr": ""}
                    last_ui_update = 0.0

                    def _consume(name: str, chunk: bytes):
                        text = chunk.decode("utf-8", errors="replace")
                        # 进度条常用 '\r'，统一转成换行，确保前端能看到每次刷新
                        text = text.replace("\r", "\n")
                        text = buffers[name] + text
                        parts = text.split("\n")
                        buffers[name] = parts[-1]
                        target = stdout_lines if name == "stdout" else stderr_lines
                        for line in parts[:-1]:
                            target.append(line + "\n")

                    while True:
                        events = sel.select(timeout=0.1)
                        for key, _ in events:
                            name = key.data
                            try:
                                chunk = key.fileobj.read(4096)
                            except Exception:
                                chunk = b""

                            if not chunk:
                                try:
                                    sel.unregister(key.fileobj)
                                except Exception:
                                    pass
                                continue

                            _consume(name, chunk)

                        # 子进程结束后，把残留 buffer 也输出出来
                        if process.poll() is not None and not sel.get_map():
                            break

                        # 限制刷新频率，避免 Streamlit 频繁重绘卡顿
                        now = time.time()
                        if (stdout_lines or stderr_lines) and (now - last_ui_update) > 0.2:
                            if stdout_lines:
                                stdout_container.code("".join(stdout_lines), language="bash")
                            if stderr_lines:
                                stderr_label_container.error("**错误 / 警告：**")
                                stderr_container.code("".join(stderr_lines), language="bash")
                            last_ui_update = now

                    # flush remaining partial lines
                    for name, rest in buffers.items():
                        if rest:
                            target = stdout_lines if name == "stdout" else stderr_lines
                            target.append(rest + "\n")

                    if stdout_lines:
                        stdout_container.code("".join(stdout_lines), language="bash")
                    if stderr_lines:
                        stderr_label_container.error("**错误 / 警告：**")
                        stderr_container.code("".join(stderr_lines), language="bash")
                    
                    return_code = process.wait()
                    
                    # Show execution result
                    if return_code == 0:
                        st.success(f"✅ 评估完成（退出码：{return_code}）")
                        eval_output_path = os.path.join(project_root, suffix_prefixed, "eval.txt")
                        if os.path.exists(eval_output_path):
                            st.info(f"📄 **评估结果已保存到：** `{suffix_prefixed}/eval.txt`")
                    else:
                        st.error(f"❌ 评估失败，退出码：{return_code}")
                        
                except Exception as e:
                    st.error(f"执行命令时出错：{e}")
            else:
                st.error(f"❌ 未找到 Conda 环境 '{conda_env}' 的 Python 可执行文件")
                st.info("请确认该环境存在，并在“设置”页重新配置。")

    # --- Plot Curves (Budget) ---
    with st.expander("Plot Curves", expanded=True):
        st.write("Select eval-complete runs (dataset + model + policy + ft) and draw budget curves.")

        eval_items = _scan_eval_complete_items()
        if not eval_items:
            st.warning("⚠️ No eval-complete runs found under front-exps/.")
            st.caption("Definition: in a folder, every target .pth has a matching *_eval.json.")
            return

        label_to_item: Dict[str, EvalRunItem] = {it.label: it for it in eval_items}
        selected_labels = st.multiselect(
            "Eval-complete runs",
            options=list(label_to_item.keys()),
            default=[],
            help="Multi-select runs that already have *_eval.json.",
        )
        selected_items = [label_to_item[lbl] for lbl in selected_labels]

        if st.button("Draw Budget Curves", use_container_width=True):
            if not selected_items:
                st.warning("Please select at least one run.")
            else:
                _plot_budget_curves(selected_items)

