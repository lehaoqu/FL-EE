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
    st.header("Generate & Monitor")
    st.session_state.setdefault("slim_ratios_text", SLIM_RATIOS_DEFAULT)
    
    # --- Part 1: Configuration ---
    with st.expander("Experiment Configuration", expanded=True):
        st.write("Configure training parameters from run_cifar_base.sh, run_speechcmds_base.sh, run_svhn_base.sh and run_glue_base.sh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model & Data Settings")
            
            dataset = st.selectbox(
                "Dataset",
                DATASET_OPTIONS,
                format_func=lambda key: DATASET_DISPLAY_NAMES.get(key, key)
            )
            st.caption("Dataset choices mirror the default scripts for CIFAR-100, SVHN, Speech Commands, and GLUE.")
            model = st.selectbox("Model (model)", ["vit", "bert"])
            slimmable = st.checkbox(
                "Enable slimmable (--slimmable)",
                value=False,
                help="Match --slimmable in utils/options.py"
            )
            slim_ratios_list = []
            slim_ratios_error = None
            if slimmable:
                slim_ratios_text = st.text_input(
                    "Slim ratios list",
                    key="slim_ratios_text",
                    value=SLIM_RATIOS_DEFAULT,
                    help="The width ratios for slimmable model; 1.0 must be the first entry."
                )
                slim_ratios_list, slim_ratios_error = _parse_slim_ratios(slim_ratios_text)
                if slim_ratios_list and abs(slim_ratios_list[0] - 1.0) > 1e-6:
                    st.warning("The first slim ratio must be 1.0 to ensure full width is included.")
                    slim_ratios_list = []
                elif slim_ratios_list:
                    st.caption("Slim ratios will be passed to --slim_ratios in the command.")
        with col2:
            st.subheader("Training Hyperparameters")
            batch_size = st.number_input("Batch Size (bs)", value=32, min_value=8, max_value=256, step=8)
            learning_rate = st.number_input("Learning Rate (lr)", value=0.05, min_value=0.001, max_value=0.1, step=0.001, format="%.4f")
            sample_ratio = st.number_input("Sample Ratio (sr)", value=0.1, min_value=0.01, max_value=1.0, step=0.05, format="%.2f")
            total_num = st.number_input("Total Clients (total_num)", value=100, min_value=10, max_value=1000, step=10)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Algorithm & Device")
            algorithm = st.selectbox("Algorithm", ["eefl", "darkflpg", "darkflpa2", "depthfl", "scalefl", "reefl"])
            device = st.text_input("Device (GPU)", value="0", help="e.g., 0 for GPU:0, or cpu")
            
        with col4:
            st.subheader("Fine-tuning & Suffix")
            fine_tuning = st.selectbox("Fine-tuning Type (ft)", ["full", "lora"])
            suffix = st.text_input("Result Suffix", value="cifar100/vit_base/noniid1000", help="Path suffix for results")
        
        st.divider()
        st.write("**Generated Command:**")
        slimmable_flag = " --slimmable" if slimmable else ""
        slim_ratios_arg = ""
        if slimmable and slim_ratios_list and not slim_ratios_error:
            slim_ratios_arg = " --slim_ratios " + " ".join(str(v) for v in slim_ratios_list)
        cmd = (
            f"python3 main.py {algorithm} {fine_tuning} --sr {sample_ratio} --total_num {total_num} --lr {learning_rate} "
            f"--bs {batch_size} --device {device} --dataset {dataset} --model {model} --suffix {suffix}{slimmable_flag}{slim_ratios_arg}"
        )
        st.code(cmd, language="bash")

    # --- Part 2: Run Training ---
    with st.expander("Run Training Script", expanded=True):
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 Using global conda environment: **{conda_env}**")
        else:
            st.caption("💡 Set conda environment in Settings page for consistent usage")
        
        if st.button("Run main.py", type="primary", use_container_width=True):
            st.info(f"Starting main.py execution with conda environment '{conda_env}'...")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Replace python3 with the full conda python path
                direct_cmd = cmd.replace("python3", python_path)
                st.info(f"Using Python: `{python_path}`")
                st.info(f"Executing: `{direct_cmd}`")
                
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
                    st.info(f"🔧 **Process ID (PID): {process.pid}** - You can manually stop it using: `kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []
                    max_lines = 100  # Limit output lines to prevent overflow
                    
                    # Read output in real-time
                    with st.expander("📋 Execution Output (Live)", expanded=True):
                        stdout_container = st.empty()
                        stderr_container = st.empty()
                        progress_info = st.empty()
                        
                        while True:
                            # Check for stop request
                            if st.session_state.get("train_stop_requested", False):
                                process.terminate()
                                st.warning("🛑 Training process terminated by user request")
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
                                    stderr_container.error("**Errors/Warnings:**")
                                    stderr_container.code(''.join(stderr_lines), language="bash")
                            
                            # Check if process finished
                            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                                break
                    
                    return_code = process.wait()
                    
                    # 显示执行结果
                    if return_code == 0:
                        st.success(f"✅ Training completed successfully (exit code: {return_code})")
                    else:
                        st.error(f"❌ Training failed with exit code: {return_code}")
                        
                except Exception as e:
                    st.error(f"Error executing command: {e}")
            else:
                st.error(f"❌ Could not find Python executable for conda environment '{conda_env}'")
                st.info("Please check if the environment exists and try setting it in Settings page.")

    # --- Part 3: Weights & Biases Integration ---
    st.divider()
    st.subheader("Weights & Biases Dashboard")
    
    # 输入 Entity 和 API Key
    col_entity, col_api = st.columns([1, 1])
    with col_entity:
        wandb_entity = st.text_input("W&B Entity (Username or Team)", value="2775257495-beihang-university", help="Your W&B username or team name")
    
    with col_api:
        wandb_api_key = st.text_input("W&B API Key", type="password", help="Found at https://wandb.ai/settings/api")
    
    # 添加登录按钮，保存 API Key 到环境变量
    col_login, col_clear = st.columns([1, 1])
    with col_login:
        if st.button("🔐 Login W&B", type="primary", use_container_width=True):
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
                            st.success(f"✅ Successfully logged in as **{user_info.username}**")
                            st.balloons()
                        else:
                            st.session_state['wandb_api_key'] = wandb_api_key
                            st.session_state['wandb_entity'] = wandb_entity
                            st.session_state['wandb_logged_in'] = True
                            st.session_state['wandb_api'] = api
                            st.success(f"✅ Successfully authenticated with W&B")
                            st.balloons()
                    except Exception as e:
                        st.error(f"Invalid API Key: {e}")
                except ImportError:
                    st.error("wandb not installed. Run: pip install wandb")
            else:
                st.warning("Please enter both W&B Entity and API Key.")
    
    with col_clear:
        if st.button("🚪 Logout", use_container_width=True):
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
            st.success("Logged out from W&B")

    # 如果已登录，显示项目选择下拉框
    if st.session_state.get('wandb_logged_in', False):
        st.info(f"✅ Logged in to W&B as entity: **{st.session_state.get('wandb_entity')}**")
        
        # 添加实时刷新控制
        col_refresh_ctrl1, col_refresh_ctrl2, col_refresh_ctrl3 = st.columns([1, 1, 1])
        with col_refresh_ctrl1:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=False, help="Automatically refresh data")
        with col_refresh_ctrl2:
            if auto_refresh:
                refresh_interval = st.slider("Refresh interval (seconds)", 10, 300, 30)
            else:
                refresh_interval = 30
        with col_refresh_ctrl3:
            if st.button("🔃 Refresh Now", type="secondary", use_container_width=True):
                st.rerun()
        
        try:
            import wandb
            import time
            
            api = st.session_state.get('wandb_api')
            if not api:
                api = wandb.Api(overrides={"api_key": st.session_state.get('wandb_api_key')})
            
            # 获取该 entity 下的所有 projects
            entity = st.session_state.get('wandb_entity')
            
            # 获取项目列表
            st.write("**Available Projects:**")
            try:
                projects = api.projects(entity=entity)
                project_names = [p.name for p in projects]
                
                if project_names:
                    selected_project = st.selectbox("Select a project", project_names)
                    
                    if selected_project:
                        st.session_state['selected_project'] = selected_project
                        
                        # 显示选中项目的 runs
                        st.divider()
                        st.subheader(f"Runs in {entity}/{selected_project}")
                        
                        # 创建 Tabs：原生数据视图 vs 网页嵌入视图
                        tab_native, tab_web = st.tabs(["📈 Native Streamlit Dashboard", "🌐 Web View"])
                        
                        # --- Tab 1: 原生视图 (使用 API 数据绘图) ---
                        with tab_native:
                            st.caption("Fetching live data directly via API (Works for private projects)")
                            
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
                                st.subheader("Training Curves (by Metric)")
                                
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
                                        st.warning(f"Could not fetch history for run {run.name}: {e}")
                                
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
                                                st.info(f"No data for metric: {metric}")
                                else:
                                    st.info("No training history found in the runs.")
                            else:
                                st.info("No runs found in this project.")
                        
                        # --- Tab 2: 网页嵌入视图 (Iframe) ---
                        with tab_web:
                            st.caption("🌐 Open W&B Dashboard in Browser")
                            # 构建项目 URL
                            project_url = f"https://wandb.ai/{entity}/{selected_project}"
                            
                            # 显示直接链接
                            col_link, col_note = st.columns([3, 2])
                            with col_link:
                                st.markdown(f"**[🔗 Open {entity}/{selected_project} on W&B →]({project_url})**")
                            with col_note:
                                st.caption("(Opens in new tab)")
                            
                            # 提供嵌入说明
                            st.info("""
                            ℹ️ **Note:** The W&B project dashboard cannot be embedded directly due to security restrictions.
                            Click the link above to view the full interactive dashboard on W&B.ai with all real-time features.
                            """)
                            
                            # 提供常用链接
                            st.subheader("Quick Links")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"[📊 Charts]({project_url}?view=charts)")
                            with col2:
                                st.markdown(f"[📈 Reports]({project_url}?view=reports)")
                            with col3:
                                st.markdown(f"[⚙️ Settings]({project_url}?view=settings)")
                
                else:
                    st.info(f"No projects found for entity: {entity}")
                    
            except Exception as e:
                st.error(f"Error fetching projects: {e}")
        
        except ImportError:
            st.error("Library `wandb` not installed. Please run `pip install wandb`.")
        except Exception as e:
            st.error(f"Connection Error: {e}")
        
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
            st.caption(f"⏱️ Page will auto-refresh every {refresh_interval} seconds...")
