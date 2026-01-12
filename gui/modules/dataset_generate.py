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


DATASET_CONFIGS = {
    "CIFAR-100": {
        "script": "generate_cifar100.py",
            "description": "生成 CIFAR-100 的默认划分（与 script/run_cifar_base.sh 保持一致）",
        "needs_test_flag": False,
    },
    "SVHN": {
        "script": "generate_svhn.py",
        "description": "生成基线实验使用的 SVHN 数据划分",
        "needs_test_flag": False,
    },
    "Speech Commands": {
        "script": "generate_speechcmd.py",
        "description": "划分 SpeechCommands 数据并生成训练集与测试集",
        "needs_test_flag": False,
    },
    "GLUE (SST-2 etc.)": {
        "script": "generate_glue.py",
        "description": "为 GLUE 任务（如 SST-2）做分词和客户端划分",
        "needs_test_flag": False,
        "glue_tasks": ["all", "sst2", "mrpc", "qqp", "qnli", "rte", "wnli"],
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_output_path(dataset_key, niid, alpha):
    """Generate the output directory path based on dataset and parameters"""
    dataset_base = {
        "CIFAR-100": "cifar100",
        "SVHN": "svhn", 
        "Speech Commands": "speechcmds",
        "GLUE (SST-2 etc.)": "glue"
    }
    
    base_name = dataset_base.get(dataset_key, "unknown")
    
    if niid and alpha is not None:
        alpha_str = str(alpha).rstrip('0').rstrip('.') if '.' in str(alpha) else str(int(alpha))
        return f"dataset/{base_name}_noniid{alpha_str}/"
    else:
        return f"dataset/{base_name}/"


def _build_command(config, niid, balance, partition, test_flag, alpha=None, glue_task=None):
    script_path = os.path.join(PROJECT_ROOT, config["script"])
    args = ["python", "-u", config["script"]]  # Unbuffered for real-time streaming
    args.append("noniid" if niid else "iid")
    args.append("balance" if balance else "unbalanced")
    args.append(partition if partition else "-")
    
    # Add alpha parameter if noniid is selected
    if niid and alpha is not None:
        args.extend(["--alpha", str(alpha)])
    
    # Add GLUE task parameter if provided
    if glue_task is not None:
        args.extend(["--task", glue_task])
    
    return args, {}


def show():
    st.header("数据集生成")
    st.write(
        "运行官方生成脚本，使 `dataset/` 下的文件结构与训练脚本的默认设置一致。"
    )

    dataset_key = st.selectbox("目标数据集", list(DATASET_CONFIGS.keys()))
    config = DATASET_CONFIGS[dataset_key]
    st.caption(config["description"])
    
    # GLUE task selection
    glue_task = None
    if dataset_key == "GLUE (SST-2 etc.)" and "glue_tasks" in config:
        glue_task = st.selectbox(
            "GLUE 任务",
            config["glue_tasks"],
            help="选择具体任务或选择 all 生成全部任务（sst2、mrpc、qqp、qnli、rte、wnli）"
        )
        if glue_task != "all":
            st.caption(f"仅生成 **{glue_task.upper()}** 任务的数据集")
        else:
            st.caption("将生成 **全部 6 个任务**：sst2、mrpc、qqp、qnli、rte、wnli")

    st.subheader("划分选项")
    col1, col2 = st.columns(2)
    with col1:
        distribution = st.selectbox("数据分布", ["iid", "noniid"])
        niid = distribution == "noniid"

        alpha = None
        if niid:
            alpha = st.selectbox(
                "Alpha（Dirichlet 浓度）", 
                [0.1, 1.0, 1000.0],
                index=2,
                help="控制非 IID 数据的异质性，值越小表示越异质。"
            )
        balance = st.checkbox("平衡各客户端间的标签（传入 'balance'）", value=True)
    with col2:
        default_partition = "dir" if niid else "pat"
        
        # Initialize session state if not exists
        if "last_distribution" not in st.session_state:
            st.session_state["last_distribution"] = distribution
            st.session_state["partition_value"] = default_partition
        
        # Update partition when distribution changes
        if st.session_state.get("last_distribution") != distribution:
            st.session_state["partition_value"] = default_partition
            st.session_state["last_distribution"] = distribution

        partition = st.text_input(
            "划分标签",
            value=st.session_state["partition_value"],
            help="使用 '-' 保持默认划分命名（脚本将 '-' 视为 None）",
        )
        # Only update session state when user actually changes the input
        if partition != st.session_state["partition_value"]:
            st.session_state["partition_value"] = partition
            
        st.caption(f"分布模式 '{distribution}' 默认使用划分标签 '{default_partition}'。")
        test_flag = False
        if config.get("needs_test_flag"):
            test_flag = st.checkbox("生成单独测试集", value=True)

    # --- Command Preview and Execution ---
    with st.expander("运行数据生成脚本", expanded=True):
        st.subheader("命令预览")
        command_args, env_vars = _build_command(config, niid, balance, partition.strip(), test_flag, alpha, glue_task)
        
        # Build command preview
        command_preview = " ".join(shlex.quote(part) for part in command_args)
        st.code(command_preview, language="bash")
        
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "searchr1")
        if conda_env != "searchr1":
            st.info(f"🐍 使用全局 Conda 环境：**{conda_env}**")
        else:
            st.caption("💡 可在“设置”页统一配置 Conda 环境")
        
        if st.button("生成数据集", type="primary", use_container_width=True):
            st.info(f"使用 Conda 环境 '{conda_env}' 开始生成数据集…")
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Replace python with the full conda python path
                direct_cmd = command_preview.replace("python", python_path)
                st.info(f"Python 解释器：`{python_path}`")
                st.info(f"执行命令：`{direct_cmd}`")
                
                # Create placeholder for real-time output
                output_placeholder = st.empty()
                error_placeholder = st.empty()
                
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
                    output_section.subheader("📋 执行输出（实时）")
                    output_section.caption("实时展示数据生成过程的标准输出与错误输出。")
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
                        output_path = _get_output_path(dataset_key, niid, alpha)
                        st.success(f"✅ 数据生成完成（退出码：{return_code}）")
                        st.info(f"📁 **输出位置：** `{output_path}`")
                        st.caption("生成文件：config.json、train/、valid/" + 
                                  (" 以及 test.pkl（取决于数据集）" if config.get("needs_test_flag") else ""))
                    else:
                        st.error(f"❌ 数据生成失败，退出码：{return_code}")
                        
                except Exception as e:
                    st.error(f"执行命令时出错：{e}")
            else:
                st.error(f"❌ 未找到 Conda 环境 '{conda_env}' 的 Python 可执行文件")
                st.info("请确认该环境存在，并在“设置”页重新配置。")
