import json
import os
import shlex
import subprocess
import selectors
import time

import streamlit as st


def _get_conda_python_path(env_name):
    """Get the Python executable path for a conda environment or system python"""
    # Handle System Python option
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


MODEL_CONFIGS = {
    "BERT-12-128": {
        "model_name": "google/bert_uncased_L-12_H-128_A-2",
        "model_key": "bert-12-128",
            "description": "12 层、隐藏维度 128、2 个注意力头的 BERT（uncased）",
        "type": "bert"
    },
    "BERT-12-256": {
        "model_name": "google/bert_uncased_L-12_H-256_A-4",
        "model_key": "bert-12-256",
        "description": "12 层、隐藏维度 256、4 个注意力头的 BERT（uncased）",
        "type": "bert"
    },
    "DeiT-Tiny": {
        "model_name": "facebook/deit-tiny-patch16-224",
        "model_key": "deit-tiny",
        "description": "数据高效的图像 Transformer（Tiny 版），patch 大小 16",
        "type": "vision"
    },
    "DeiT-Small": {
        "model_name": "facebook/deit-small-patch16-224",
        "model_key": "deit-small",
        "description": "数据高效的图像 Transformer（Small 版），patch 大小 16",
        "type": "vision"
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show():
    st.header("模型下载")
    st.write(
        "从 HuggingFace Hub 下载预训练模型，默认保存到 ./models/ 目录。"
    )

    # Model selection
    st.subheader("选择要下载的模型")
    
    selected_models = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**BERT 模型**")
        for key in ["BERT-12-128", "BERT-12-256"]:
            config = MODEL_CONFIGS[key]
            if st.checkbox(
                f"{key}",
                key=f"check_{key}",
                help=config["description"]
            ):
                selected_models.append(key)
    
    with col2:
        st.write("**视觉模型**")
        for key in ["DeiT-Tiny", "DeiT-Small"]:
            config = MODEL_CONFIGS[key]
            if st.checkbox(
                f"{key}",
                key=f"check_{key}",
                help=config["description"]
            ):
                selected_models.append(key)

    # Download options
    st.subheader("下载选项")
    save_dir = st.text_input(
        "保存目录",
        value="./models",
        help="模型保存路径（相对项目根目录）"
    )

    # --- Command Preview and Execution ---
    with st.expander("下载已选择的模型", expanded=True):
        if not selected_models:
            st.warning("⚠️ 未选择模型，请至少选择一个模型后再下载。")
            return
        
        st.subheader("已选模型")
        for model_key in selected_models:
            config = MODEL_CONFIGS[model_key]
            st.write(f"- **{model_key}**: `{config['model_name']}`")
        
        st.subheader("命令预览")
        # Build command with selected models
        command_args = ["python", "download_models.py"]
        command_args.append("--models")
        command_args.extend([MODEL_CONFIGS[key]["model_key"] for key in selected_models])
        if save_dir != "./models":
            command_args.extend(["--save-dir", save_dir])
        
        command_preview = " ".join(shlex.quote(part) for part in command_args)
        st.code(command_preview, language="bash")
        
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 使用全局 Conda 环境：**{conda_env}**")
        else:
            st.caption("💡 可在“设置”页统一配置 Conda 环境")
        
        if st.button("开始下载", type="primary", use_container_width=True):
            st.info(f"使用 Conda 环境 '{conda_env}' 开始下载模型…")
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Build command with direct python path and selected models
                model_args = " ".join(MODEL_CONFIGS[key]["model_key"] for key in selected_models)
                direct_cmd = f"{python_path} -u download_models.py --models {model_args}"
                if save_dir != "./models":
                    direct_cmd += f" --save-dir {shlex.quote(save_dir)}"
                
                st.info(f"Python 解释器：`{python_path}`")
                st.info(f"执行命令：`{direct_cmd}`")
                
                try:
                    # Force unbuffered output for real-time streaming
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"

                    # Execute command with real-time output (non-blocking)
                    process = subprocess.Popen(
                        direct_cmd,
                        shell=True,
                        cwd=PROJECT_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                        env=env,
                    )
                    
                    # 显示进程号
                    st.info(f"🔧 **进程号 (PID): {process.pid}** - 如需手动停止可执行：`kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []

                    output_section = st.container()
                    output_section.subheader("📋 下载输出（实时）")
                    output_section.caption("实时展示下载过程的标准输出与错误输出。")
                    stdout_container = output_section.empty()
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

                        if process.poll() is not None and not sel.get_map():
                            break

                        now = time.time()
                        if (stdout_lines or stderr_lines) and (now - last_ui_update) > 0.2:
                            if stdout_lines:
                                stdout_container.code("".join(stdout_lines), language="bash")
                            if stderr_lines:
                                stderr_container.error("**错误 / 警告：**")
                                stderr_container.code("".join(stderr_lines), language="bash")
                            last_ui_update = now

                    for name, rest in buffers.items():
                        if rest:
                            target = stdout_lines if name == "stdout" else stderr_lines
                            target.append(rest + "\n")

                    if stdout_lines:
                        stdout_container.code("".join(stdout_lines), language="bash")
                    if stderr_lines:
                        stderr_container.error("**错误 / 警告：**")
                        stderr_container.code("".join(stderr_lines), language="bash")
                    
                    return_code = process.wait()
                    
                    # Show execution result
                    if return_code == 0:
                        st.success(f"✅ 模型下载完成（退出码：{return_code}）")
                        st.info(f"📁 **模型已保存到：** `{os.path.join(PROJECT_ROOT, save_dir)}`")
                        st.caption("可以在训练脚本中直接使用这些模型。")
                    else:
                        st.error(f"❌ 模型下载失败，退出码：{return_code}")
                        
                except Exception as e:
                    st.error(f"执行命令时出错：{e}")
            else:
                st.error(f"❌ 未找到 Conda 环境 '{conda_env}' 的 Python 可执行文件")
                st.info("请确认该环境存在，并在“设置”页重新配置。")
