import json
import os
import shlex
import subprocess

import streamlit as st


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


DATASET_CONFIGS = {
    "CIFAR-100": {
        "script": "generate_cifar100.py",
        "description": "Generates CIFAR-100 splits (the same defaults from script/run_cifar_base.sh).",
        "needs_test_flag": False,
    },
    "SVHN": {
        "script": "generate_svhn.py",
        "description": "Produces SVHN splits used by the base runs.",
        "needs_test_flag": False,
    },
    "Speech Commands": {
        "script": "generate_speechcmd.py",
        "description": "Partitions SpeechCommands data and generates both train and test sets automatically.",
        "needs_test_flag": False,
    },
    "GLUE (SST-2 etc.)": {
        "script": "generate_glue.py",
        "description": "Prepares GLUE tasks such as SST-2 by tokenizing and splitting into clients.",
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
    args = ["python3", config["script"]]  # Use relative script path for conda run
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
    st.header("Dataset Generation")
    st.write(
        "Run the official generation scripts so the dataset folders under `dataset/` match the defaults used by the training runs."
    )

    dataset_key = st.selectbox("Target dataset", list(DATASET_CONFIGS.keys()))
    config = DATASET_CONFIGS[dataset_key]
    st.caption(config["description"])
    
    # GLUE task selection
    glue_task = None
    if dataset_key == "GLUE (SST-2 etc.)" and "glue_tasks" in config:
        glue_task = st.selectbox(
            "GLUE Task",
            config["glue_tasks"],
            help="Select specific task or 'all' to generate all tasks (sst2, mrpc, qqp, qnli, rte, wnli)"
        )
        if glue_task != "all":
            st.caption(f"Will generate only **{glue_task.upper()}** dataset")
        else:
            st.caption("Will generate **all 6 tasks**: sst2, mrpc, qqp, qnli, rte, wnli")

    st.subheader("Partition Options")
    col1, col2 = st.columns(2)
    with col1:
        distribution = st.selectbox("Distribution", ["iid", "noniid"])
        niid = distribution == "noniid"

        alpha = None
        if niid:
            alpha = st.selectbox(
                "Alpha (Dirichlet concentration)", 
                [0.1, 1.0, 1000.0],
                index=2,
                help="Controls non-IID data heterogeneity. Lower values = more heterogeneous."
            )
        balance = st.checkbox("Balance labels across clients (pass 'balance')", value=True)
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
            "Partition tag",
            value=st.session_state["partition_value"],
            help="Use '-' to keep the existing partition naming (script treats '-' as None).",
        )
        # Only update session state when user actually changes the input
        if partition != st.session_state["partition_value"]:
            st.session_state["partition_value"] = partition
            
        st.caption(f"Distribution '{distribution}' defaults to partition '{default_partition}'.")
        test_flag = False
        if config.get("needs_test_flag"):
            test_flag = st.checkbox("Generate explicit test split", value=True)

    # --- Command Preview and Execution ---
    with st.expander("Run Dataset Generation Script", expanded=True):
        st.subheader("Command Preview")
        command_args, env_vars = _build_command(config, niid, balance, partition.strip(), test_flag, alpha, glue_task)
        
        # Build command preview
        command_preview = " ".join(shlex.quote(part) for part in command_args)
        st.code(command_preview, language="bash")
        
        # Get conda environment from global settings
        conda_env = st.session_state.get("conda_env", "fl-ee")
        if conda_env != "fl-ee":
            st.info(f"🐍 Using global conda environment: **{conda_env}**")
        else:
            st.caption("💡 Set conda environment in Settings page for consistent usage")
        
        if st.button("Generate dataset", type="primary", use_container_width=True):
            st.info(f"Starting dataset generation with conda environment '{conda_env}'...")
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Replace python3 with the full conda python path
                direct_cmd = command_preview.replace("python3", python_path)
                st.info(f"Using Python: `{python_path}`")
                st.info(f"Executing: `{direct_cmd}`")
                
                # Create placeholder for real-time output
                output_placeholder = st.empty()
                error_placeholder = st.empty()
                
                try:
                    # Execute command with real-time output
                    import subprocess
                    process = subprocess.Popen(
                        direct_cmd,
                        shell=True,
                        cwd=PROJECT_ROOT,
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
                    max_lines = 100  # Limit output lines to prevent overflow
                    
                    # Read output in real-time (avoid nested expanders inside the run expander)
                    output_section = st.container()
                    output_section.subheader("📋 Execution Output (Live)")
                    stdout_container = output_section.empty()
                    stderr_container = output_section.empty()
                    progress_info = output_section.empty()

                    while True:
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
                    
                    # Show execution result
                    if return_code == 0:
                        output_path = _get_output_path(dataset_key, niid, alpha)
                        st.success(f"✅ Dataset generation completed successfully (exit code: {return_code})")
                        st.info(f"📁 **Output saved to:** `{output_path}`")
                        st.caption("Files generated: config.json, train/ folder, valid/ folder" + 
                                  (" and test.pkl (for some datasets)" if config.get("needs_test_flag") else ""))
                    else:
                        st.error(f"❌ Dataset generation failed with exit code: {return_code}")
                        
                except Exception as e:
                    st.error(f"Error executing command: {e}")
            else:
                st.error(f"❌ Could not find Python executable for conda environment '{conda_env}'")
                st.info("Please check if the environment exists and try setting it in Settings page.")
