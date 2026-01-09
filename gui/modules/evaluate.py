import streamlit as st
import os
import json
import subprocess


def _get_conda_python_path(env_name):
    """Get the Python executable path for a conda environment or system python"""
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


def _scan_exps_suffixes():
    """Recursively scan EXPS directory for available result suffixes"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exps_dir = os.path.join(project_root, "EXPS")
    
    suffixes = []
    
    if not os.path.exists(exps_dir):
        return suffixes
    
    # Walk through EXPS directory to find all subdirectories
    for root, dirs, files in os.walk(exps_dir):
        # Get relative path from EXPS
        rel_path = os.path.relpath(root, exps_dir)
        if rel_path != '.':
            # Check if this directory contains model files (.pth)
            has_models = any(f.endswith('.pth') for f in files)
            if has_models:
                suffixes.append(rel_path)
    
    return sorted(suffixes)


def show():
    """显示 Evaluate 页面"""
    st.header("Evaluate")
    st.write("Run evaluation on trained models from EXPS directory.")
    
    # --- Configuration ---
    with st.expander("Evaluation Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model & Data Settings")
            
            # Scan available suffixes
            available_suffixes = _scan_exps_suffixes()
            
            if available_suffixes:
                st.caption(f"📁 Found {len(available_suffixes)} experiment(s) with trained models")
                suffix = st.selectbox(
                    "Experiment Suffix",
                    available_suffixes,
                    help="Auto-detected from EXPS directory"
                )
            else:
                st.warning("⚠️ No trained models found in EXPS/ directory.")
                suffix = st.text_input(
                    "Experiment Suffix (manual input)",
                    value="test/cifar100/vit_base",
                    help="Manually specify experiment suffix"
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
                    "Dataset",
                    sorted(available_datasets),
                    help="Dataset to evaluate on"
                )
            else:
                dataset = st.text_input("Dataset", value="cifar100_noniid1000")
            
            model = st.selectbox("Model", ["vit", "bert"])
        
        with col2:
            st.subheader("Evaluation Settings")
            
            algorithm = st.selectbox("Algorithm", ["eefl", "darkflpg", "darkflpa2", "depthfl", "scalefl", "reefl"])
            boosted = st.selectbox("Boosted Mode", ["boosted", "unboosted"])
            
            valid_ratio = st.number_input(
                "Validation Ratio",
                value=0.2,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                help="Ratio of data to use for validation"
            )
            
            if_mode = st.selectbox(
                "Inference Mode",
                ["all", "anytime", "budgeted"],
                help="all: both anytime and budgeted, anytime: early exit at any layer, budgeted: with budget constraints"
            )
            
            device = st.text_input("Device (GPU)", value="0", help="e.g., 0 for GPU:0, or cpu")
            
            fine_tuning = st.selectbox("Fine-tuning Type", ["full", "lora"])
        
        st.divider()
        st.write("**Generated Command:**")
        cmd = (
            f"python3 eval.py {algorithm} {boosted} --suffix {suffix} --device {device} "
            f"--dataset {dataset} --model {model} --valid_ratio {valid_ratio} --if_mode {if_mode} --ft {fine_tuning}"
        )
        st.code(cmd, language="bash")
    
    # --- Run Evaluation ---
    with st.expander("Run Evaluation", expanded=True):
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 Using global conda environment: **{conda_env}**")
        else:
            st.caption("💡 Set conda environment in Settings page for consistent usage")
        
        if st.button("Run Evaluation", type="primary", use_container_width=True):
            st.info(f"Starting evaluation with conda environment '{conda_env}'...")
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
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
                    
                    # Display process ID
                    st.info(f"🔧 **Process ID (PID): {process.pid}** - You can manually stop it using: `kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []
                    max_lines = 100
                    
                    # Read output in real-time
                    with st.expander("📋 Evaluation Output (Live)", expanded=True):
                        stdout_container = st.empty()
                        stderr_container = st.empty()
                        progress_info = st.empty()
                        
                        while True:
                            # Read stdout
                            stdout_line = process.stdout.readline()
                            if stdout_line:
                                is_progress = any(indicator in stdout_line for indicator in ['%|', 'it/s]', 'iB/s]', 'B/s]'])
                                
                                if is_progress:
                                    progress_info.info(f"⏳ {stdout_line.strip()}")
                                    if stdout_lines and any(ind in stdout_lines[-1] for ind in ['%|', 'it/s]', 'iB/s]', 'B/s]']):
                                        stdout_lines[-1] = stdout_line
                                    else:
                                        stdout_lines.append(stdout_line)
                                else:
                                    stdout_lines.append(stdout_line)
                                
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
                    
                    # Show execution result
                    if return_code == 0:
                        st.success(f"✅ Evaluation completed successfully (exit code: {return_code})")
                        eval_output_path = os.path.join(project_root, suffix, "eval.txt")
                        if os.path.exists(eval_output_path):
                            st.info(f"📄 **Evaluation results saved to:** `{suffix}/eval.txt`")
                    else:
                        st.error(f"❌ Evaluation failed with exit code: {return_code}")
                        
                except Exception as e:
                    st.error(f"Error executing command: {e}")
            else:
                st.error(f"❌ Could not find Python executable for conda environment '{conda_env}'")
                st.info("Please check if the environment exists and try setting it in Settings page.")

