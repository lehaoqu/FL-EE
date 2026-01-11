import json
import os
import re
import subprocess

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components





def _get_conda_python_path(env_name):
    """Get the Python executable path for a conda environment or system python"""
    # Handle System Python option
    if env_name == "System Python (python3)":
        return "/usr/bin/python3"
    
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


def _scan_available_datasets():
    """Scan dataset directory for available datasets with config.json"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, "dataset")
    
    available_datasets = []
    
    if not os.path.exists(dataset_dir):
        return available_datasets
    
    # Scan for directories with config.json
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # Check if config.json exists directly in this folder
            if os.path.exists(os.path.join(item_path, "config.json")):
                available_datasets.append(item)
            else:
                # Check subdirectories (e.g., glue/sst2/config.json)
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path) and os.path.exists(os.path.join(subitem_path, "config.json")):
                        available_datasets.append(f"{item}/{subitem}")
    
    return sorted(available_datasets)


DATASET_OPTIONS = [
    "cifar100_noniid1000",
    "cifar100_noniid1",
    "cifar100_noniid0.1",
    "svhn",
    "speechcmds",
    "sst2",
]

DATASET_DISPLAY_NAMES = {
    "cifar100_noniid1000": "CIFAR-100 (noniid 1000)",
    "cifar100_noniid1": "CIFAR-100 (noniid 1)",
    "cifar100_noniid0.1": "CIFAR-100 (noniid 0.1)",
    "svhn": "SVHN",
    "speechcmds": "Speech Commands",
    "sst2": "GLUE SST-2",
}


SLIM_RATIOS_DEFAULT = "[1.0, 0.75, 0.5, 0.25]"


def _parse_slim_ratios(raw_text: str):
    stripped = (raw_text or "").strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    if not stripped:
        return [], "请输入至少一个比例，例如 [1.0, 0.75, 0.5, 0.25]"

    parts = re.split(r"[\s,]+", stripped)
    ratios = []
    for part in parts:
        if not part:
            continue
        try:
            ratios.append(float(part))
        except ValueError:
            return [], f"无法解析比例: {part}"

    if not ratios:
        return [], "列表中必须包含至少一个数值"

    return ratios, None


def show():
    """显示 Generate & Monitor 页面"""
    st.header("训练与监控")
    st.session_state.setdefault("slim_ratios_text", SLIM_RATIOS_DEFAULT)
    
    # --- Part 1: Configuration ---
    with st.expander("实验配置", expanded=True):
        st.write("配置与 run_cifar_base.sh、run_speechcmds_base.sh、run_svhn_base.sh、run_glue_base.sh 一致的训练参数。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模型与数据设置")
            
            # Scan available datasets
            available_datasets = _scan_available_datasets()
            
            if available_datasets:
                st.caption(f"📁 在 dataset/ 目录中发现 {len(available_datasets)} 个数据集")
                dataset = st.selectbox(
                    "数据集",
                    available_datasets,
                    help="自动检测到包含 config.json 的数据集"
                )
            else:
                st.warning("⚠️ dataset/ 目录中未发现数据集，请先生成数据集。")
                dataset = st.text_input(
                    "数据集（手动输入）",
                    value="cifar100_noniid1000",
                    help="手动指定数据集名称"
                )
            
            model = st.selectbox("模型 (model)", ["vit", "bert"])
            slimmable = st.checkbox(
                "启用 slimmable (--slimmable)",
                value=False,
                help="对应 utils/options.py 中的 --slimmable 选项"
            )
            slim_ratios_list = []
            slim_ratios_error = None
            if slimmable:
                slim_ratios_text = st.text_input(
                    "Slim 比例列表",
                    key="slim_ratios_text",
                    value=SLIM_RATIOS_DEFAULT,
                    help="slimmable 模型的宽度比例列表；首个值必须为 1.0"
                )
                slim_ratios_list, slim_ratios_error = _parse_slim_ratios(slim_ratios_text)
                if slim_ratios_list and abs(slim_ratios_list[0] - 1.0) > 1e-6:
                    st.warning("首个 Slim 比例必须为 1.0，确保包含全宽模型。")
                    slim_ratios_list = []
                elif slim_ratios_list:
                    st.caption("Slim 比例将作为 --slim_ratios 传入命令。")
        with col2:
            st.subheader("训练超参数")
            batch_size = st.number_input("批大小 (bs)", value=32, min_value=8, max_value=256, step=8)
            learning_rate = st.number_input("学习率 (lr)", value=0.05, min_value=0.001, max_value=0.1, step=0.001, format="%.4f")
            sample_ratio = st.number_input("采样比例 (sr)", value=0.1, min_value=0.01, max_value=1.0, step=0.05, format="%.2f")
            total_num = st.number_input("客户端总数 (total_num)", value=100, min_value=10, max_value=1000, step=10)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("算法与设备")
            algorithm = st.selectbox("算法", ["eefl", "darkflpg", "darkflpa2", "depthfl", "scalefl", "reefl"])
            device = st.text_input("设备 (GPU)", value="0", help="例如 0 表示 GPU:0，或填 cpu")
            
        with col4:
            st.subheader("微调与结果后缀")
            fine_tuning = st.selectbox("微调方式 (ft)", ["full", "lora"])
            
            # Generate suggested suffix based on dataset and model
            dataset_base = dataset.split('_')[0] if '_' in dataset else dataset
            suggested_suffix = f"{dataset_base}/{model}_base/{dataset.replace(dataset_base + '_', '')}" if '_' in dataset else f"{dataset_base}/{model}_base/iid"
            
            suffix = st.text_input(
                "结果后缀", 
                value=suggested_suffix,
                help="结果路径后缀，默认根据数据集与模型生成。"
            )
            st.caption(f"💡 推荐：`{suggested_suffix}`")
        
        st.divider()
        st.write("**生成的命令：**")
        slimmable_flag = " --slimmable" if slimmable else ""
        slim_ratios_arg = ""
        if slimmable and slim_ratios_list and not slim_ratios_error:
            slim_ratios_arg = " --slim_ratios " + " ".join(str(v) for v in slim_ratios_list)
        
        # Add "front-exps/" prefix to suffix
        full_suffix = f"front-exps/{suffix}" if suffix and not suffix.startswith("front-exps/") else suffix
        
        cmd = (
            f"python3 main.py {algorithm} {fine_tuning} --sr {sample_ratio} --total_num {total_num} --lr {learning_rate} "
            f"--bs {batch_size} --device {device} --dataset {dataset} --model {model} --suffix {full_suffix}{slimmable_flag}{slim_ratios_arg}"
        )
        st.code(cmd, language="bash")

    # --- Part 2: Run Training ---
    with st.expander("运行训练脚本", expanded=True):
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 使用全局 Conda 环境：**{conda_env}**")
        else:
            st.caption("💡 可在“设置”页统一配置 Conda 环境")
        
        if st.button("运行 main.py", type="primary", use_container_width=True):
            st.info(f"使用 Conda 环境 '{conda_env}' 开始执行 main.py…")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Replace python3 with the full conda python path
                direct_cmd = cmd.replace("python3", python_path)
                st.info(f"Python 解释器：`{python_path}`")
                st.info(f"执行命令：`{direct_cmd}`")
                
                try:
                    # Execute command with real-time output
                    process = subprocess.Popen(
                        direct_cmd,
                        shell=True,
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Store process in session state
                    st.session_state["train_process"] = process
                    st.session_state["train_process_running"] = True
                    
                    # Display process ID
                    st.info(f"🔧 **进程号 (PID): {process.pid}** - 如需手动停止可执行：`kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []
                    max_lines = 100  # Limit output lines to prevent overflow
                    
                    # Read output in real-time
                    with st.expander("📋 执行输出（实时）", expanded=True):
                        stdout_container = st.empty()
                        stderr_container = st.empty()
                        progress_info = st.empty()
                        
                        while True:
                            # Check for stop request
                            if st.session_state.get("train_stop_requested", False):
                                process.terminate()
                                st.warning("🛑 已按用户请求终止训练进程")
                                break
                            
                            # Read stdout
                            stdout_line = process.stdout.readline()
                            if stdout_line:
                                # Check if this is a progress line
                                is_progress = any(indicator in stdout_line for indicator in ['%|', 'it/s]', 'iB/s]', 'B/s]', 'Downloading'])
                                
                                if is_progress:
                                    # Show progress in separate area
                                    progress_info.info(f"⏳ {stdout_line.strip()}")
                                    # Only keep last progress line in main output
                                    if stdout_lines and any(ind in stdout_lines[-1] for ind in ['%|', 'it/s]', 'iB/s]', 'B/s]']):
                                        stdout_lines[-1] = stdout_line
                                    else:
                                        stdout_lines.append(stdout_line)
                                else:
                                    stdout_lines.append(stdout_line)
                                
                                # Keep only last max_lines
                                if len(stdout_lines) > max_lines:
                                    stdout_lines = stdout_lines[-max_lines:]
                                
                                stdout_container.code(''.join(stdout_lines), language="bash")
                            
                            # Read stderr
                            stderr_line = process.stderr.readline()
                            if stderr_line:
                                stderr_lines.append(stderr_line)
                                if len(stderr_lines) > max_lines:
                                    stderr_lines = stderr_lines[-max_lines:]
                                if stderr_lines:
                                    stderr_container.error("**错误 / 警告：**")
                                    stderr_container.code(''.join(stderr_lines), language="bash")
                            
                            # Check if process finished
                            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                                break
                    
                    return_code = process.wait()
                    
                    # 显示执行结果
                    if return_code == 0:
                        st.success(f"✅ 训练完成（退出码：{return_code}）")
                    else:
                        st.error(f"❌ 训练失败，退出码：{return_code}")
                        
                except Exception as e:
                    st.error(f"执行命令时出错：{e}")
            else:
                st.error(f"❌ 未找到 Conda 环境 '{conda_env}' 的 Python 可执行文件")
                st.info("请确认该环境存在，并在“设置”页重新配置。")

    # --- Part 3: Weights & Biases Integration ---
    st.divider()
    st.subheader("Weights & Biases 仪表盘")
    
    # 输入 Entity 和 API Key
    col_entity, col_api = st.columns([1, 1])
    with col_entity:
        wandb_entity = st.text_input("W&B 实体（用户名或团队）", value="2775257495-beihang-university", help="填写你的 W&B 用户名或团队名")
    
    with col_api:
        wandb_api_key = st.text_input("W&B API Key", type="password", help="可在 https://wandb.ai/settings/api 获取")
    
    # 添加登录按钮，保存 API Key 到环境变量
    col_login, col_clear = st.columns([1, 1])
    with col_login:
        if st.button("🔐 登录 W&B", type="primary", use_container_width=True):
            if wandb_api_key and wandb_entity:
                try:
                    # 直接设置环境变量和 wandb 配置
                    os.environ['WANDB_API_KEY'] = wandb_api_key
                    
                    import wandb
                    # 通过 API 对象初始化来验证 key
                    api = wandb.Api(overrides={"api_key": wandb_api_key})
                    
                    # 简单的验证：尝试获取当前用户信息
                    try:
                        user_info = api.viewer
                        if user_info and hasattr(user_info, 'username'):
                            st.session_state['wandb_api_key'] = wandb_api_key
                            st.session_state['wandb_entity'] = wandb_entity
                            st.session_state['wandb_logged_in'] = True
                            st.session_state['wandb_api'] = api
                            st.success(f"✅ 已登录，当前用户：**{user_info.username}**")
                            st.balloons()
                        else:
                            st.session_state['wandb_api_key'] = wandb_api_key
                            st.session_state['wandb_entity'] = wandb_entity
                            st.session_state['wandb_logged_in'] = True
                            st.session_state['wandb_api'] = api
                            st.success(f"✅ W&B 认证成功")
                            st.balloons()
                    except Exception as e:
                        st.error(f"API Key 无效：{e}")
                except ImportError:
                    st.error("未安装 wandb，请运行：pip install wandb")
            else:
                st.warning("请同时填写 W&B 实体与 API Key。")
    
    with col_clear:
        if st.button("🚪 退出登录", use_container_width=True):
            if 'wandb_api_key' in st.session_state:
                del st.session_state['wandb_api_key']
            if 'wandb_entity' in st.session_state:
                del st.session_state['wandb_entity']
            if 'wandb_logged_in' in st.session_state:
                del st.session_state['wandb_logged_in']
            if 'wandb_api' in st.session_state:
                del st.session_state['wandb_api']
            if 'WANDB_API_KEY' in os.environ:
                del os.environ['WANDB_API_KEY']
            st.success("已从 W&B 登出")

    # 如果已登录，显示项目选择下拉框
    if st.session_state.get('wandb_logged_in', False):
        st.info(f"✅ 已登录 W&B，实体：**{st.session_state.get('wandb_entity')}**")
        
        # 添加实时刷新控制
        col_refresh_ctrl1, col_refresh_ctrl2, col_refresh_ctrl3 = st.columns([1, 1, 1])
        with col_refresh_ctrl1:
            auto_refresh = st.checkbox("🔄 自动刷新", value=False, help="自动刷新数据")
        with col_refresh_ctrl2:
            if auto_refresh:
                refresh_interval = st.slider("刷新间隔（秒）", 10, 300, 30)
            else:
                refresh_interval = 30
        with col_refresh_ctrl3:
            if st.button("🔃 立即刷新", type="secondary", use_container_width=True):
                st.rerun()
        
        try:
            import wandb
            import time
            
            api = st.session_state.get('wandb_api')
            if not api:
                api = wandb.Api(overrides={"api_key": st.session_state.get('wandb_api_key')})
            
            # 获取该实体下的所有 projects
            entity = st.session_state.get('wandb_entity')
            
            # 获取项目列表
            st.write("**可用项目：**")
            try:
                projects = api.projects(entity=entity)
                project_names = [p.name for p in projects]
                
                if project_names:
                    selected_project = st.selectbox("选择项目", project_names)
                    
                    if selected_project:
                        st.session_state['selected_project'] = selected_project
                        
                        # 显示选中项目的 runs
                        st.divider()
                        st.subheader(f"{entity}/{selected_project} 的运行记录")
                        
                        # 创建 Tabs：原生数据视图 vs 网页嵌入视图
                        tab_native, tab_web = st.tabs(["📈 原生数据面板", "🌐 网页视图"])
                        
                        # --- Tab 1: 原生视图 (使用 API 数据绘图) ---
                        with tab_native:
                            st.caption("通过 API 实时拉取数据（私有项目同样适用）")
                            
                            # 获取该项目的 Run 列表
                            runs = list(api.runs(f"{entity}/{selected_project}", per_page=10, order="-created_at"))
                            
                            if runs:
                                run_data = []
                                # 收集所有 runs 的基本信息
                                for run in runs:
                                    run_info = {
                                        "Name": run.name,
                                        "State": run.state,
                                        "ID": run.id,
                                        "Created": run.created_at,
                                    }
                                    # 合并 summary metrics - 安全处理
                                    try:
                                        if hasattr(run, 'summary') and run.summary:
                                            summary_dict = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
                                            if isinstance(summary_dict, dict):
                                                run_info.update(summary_dict)
                                    except Exception as e:
                                        pass  # 忽略 summary 错误
                                    
                                    run_data.append(run_info)
                                
                                df_runs = pd.DataFrame(run_data)
                                
                                # 显示运行列表表格
                                st.dataframe(df_runs, use_container_width=True)
                                
                                # 按 Metric 分别绘图，展示所有 runs 的历史曲线
                                st.subheader("训练曲线（按指标）")
                                
                                # 收集所有 runs 的历史数据，找出所有 metrics
                                all_metrics = set()
                                runs_history = {}
                                
                                for run in runs:
                                    try:
                                        history = run.history(samples=500)
                                        if not history.empty:
                                            runs_history[run.name] = history
                                            # 收集该 run 中的所有指标列
                                            all_metrics.update(history.columns)
                                    except Exception as e:
                                        st.warning(f"无法获取运行 {run.name} 的历史：{e}")
                                
                                # 移除非 metric 的列 (Step, Epoch 等)
                                metric_cols = [col for col in all_metrics if col not in ['Step', '_step', 'epoch', '_timestamp']]
                                
                                if metric_cols and runs_history:
                                    # 为每个 metric 创建一张图
                                    cols_per_row = 2
                                    num_metrics = len(metric_cols)
                                    
                                    for idx, metric in enumerate(sorted(metric_cols)):
                                        # 每两张图创建一行
                                        if idx % cols_per_row == 0:
                                            metric_cols_layout = st.columns(cols_per_row)
                                        
                                        col_idx = idx % cols_per_row
                                        with metric_cols_layout[col_idx]:
                                            st.write(f"**{metric}**")
                                            
                                            # 收集该 metric 的所有 runs 数据
                                            metric_chart_data = {}
                                            has_data = False
                                            
                                            for run_name, history in runs_history.items():
                                                if metric in history.columns:
                                                    # 提取该 run 该 metric 的数据
                                                    metric_data = history[[metric]].dropna()
                                                    if not metric_data.empty:
                                                        metric_chart_data[run_name] = metric_data[metric].values
                                                        has_data = True
                                            
                                            if has_data:
                                                # 创建多条线的图表
                                                import altair as alt
                                                
                                                # 准备数据：将多条线的数据转换为 long format
                                                chart_data_list = []
                                                for run_name, values in metric_chart_data.items():
                                                    for step, value in enumerate(values):
                                                        chart_data_list.append({
                                                            'Step': step,
                                                            'Value': value,
                                                            'Run': run_name
                                                        })
                                                
                                                if chart_data_list:
                                                    chart_df = pd.DataFrame(chart_data_list)
                                                    
                                                    # 使用 Altair 绘制多条线
                                                    chart = alt.Chart(chart_df).mark_line().encode(
                                                        x='Step:Q',
                                                        y='Value:Q',
                                                        color='Run:N',
                                                        tooltip=['Step', 'Value', 'Run']
                                                    ).properties(
                                                        width=400,
                                                        height=300
                                                    )
                                                    
                                                    st.altair_chart(chart, use_container_width=True)
                                            else:
                                                st.info(f"该指标暂无数据：{metric}")
                                else:
                                    st.info("这些运行中没有可用的训练历史。")
                            else:
                                st.info("该项目下尚无运行记录。")
                        
                        # --- Tab 2: 网页嵌入视图 (Iframe) ---
                        with tab_web:
                            st.caption("🌐 在浏览器中打开 W&B 仪表盘")
                            # 构建项目 URL
                            project_url = f"https://wandb.ai/{entity}/{selected_project}"
                            
                            # 显示直接链接
                            col_link, col_note = st.columns([3, 2])
                            with col_link:
                                st.markdown(f"**[🔗 在 W&B 打开 {entity}/{selected_project} →]({project_url})**")
                            with col_note:
                                st.caption("（将在新标签页打开）")
                            
                            # 提供嵌入说明
                            st.info("""
                            ℹ️ 提示：受安全限制，W&B 仪表盘无法直接内嵌。
                            请点击上方链接，在 W&B.ai 中查看完整的交互式仪表盘。
                            """)
                            
                            # 提供常用链接
                            st.subheader("快捷入口")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"[📊 图表]({project_url}?view=charts)")
                            with col2:
                                st.markdown(f"[📈 报告]({project_url}?view=reports)")
                            with col3:
                                st.markdown(f"[⚙️ 设置]({project_url}?view=settings)")
                
                else:
                    st.info(f"实体 {entity} 下没有项目。")
                    
            except Exception as e:
                st.error(f"获取项目时出错：{e}")
        
        except ImportError:
            st.error("未安装 wandb 库，请运行：pip install wandb。")
        except Exception as e:
            st.error(f"连接错误：{e}")
        
        # 实现自动刷新逻辑
        if auto_refresh:
            # 使用 JavaScript 计时器来定时刷新
            components.html(f"""
            <script>
                // 每隔指定秒数刷新页面
                setTimeout(function() {{
                    location.reload();
                }}, {refresh_interval * 1000});
            </script>
            """)
            st.caption(f"⏱️ 页面将每 {refresh_interval} 秒自动刷新一次")
