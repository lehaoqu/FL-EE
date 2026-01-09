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
        "description": "BERT with 12 layers, hidden size 128, 2 attention heads, uncased",
        "type": "bert"
    },
    "BERT-12-256": {
        "model_name": "google/bert_uncased_L-12_H-256_A-4",
        "model_key": "bert-12-256",
        "description": "BERT with 12 layers, hidden size 256, 4 attention heads, uncased",
        "type": "bert"
    },
    "DeiT-Tiny": {
        "model_name": "facebook/deit-tiny-patch16-224",
        "model_key": "deit-tiny",
        "description": "Data-efficient Image Transformer (Tiny version) with patch size 16",
        "type": "vision"
    },
    "DeiT-Small": {
        "model_name": "facebook/deit-small-patch16-224",
        "model_key": "deit-small",
        "description": "Data-efficient Image Transformer (Small version) with patch size 16",
        "type": "vision"
    },
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show():
    st.header("Model Download")
    st.write(
        "Download pre-trained models from HuggingFace Hub. Models will be saved to the `./models/` directory."
    )

    # Model selection
    st.subheader("Select Models to Download")
    
    selected_models = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**BERT Models**")
        for key in ["BERT-12-128", "BERT-12-256"]:
            config = MODEL_CONFIGS[key]
            if st.checkbox(
                f"{key}",
                key=f"check_{key}",
                help=config["description"]
            ):
                selected_models.append(key)
    
    with col2:
        st.write("**Vision Models**")
        for key in ["DeiT-Tiny", "DeiT-Small"]:
            config = MODEL_CONFIGS[key]
            if st.checkbox(
                f"{key}",
                key=f"check_{key}",
                help=config["description"]
            ):
                selected_models.append(key)

    # Download options
    st.subheader("Download Options")
    save_dir = st.text_input(
        "Save Directory",
        value="./models",
        help="Directory where models will be saved (relative to project root)"
    )

    # --- Command Preview and Execution ---
    with st.expander("Download Selected Models", expanded=True):
        if not selected_models:
            st.warning("⚠️ No models selected. Please select at least one model to download.")
            return
        
        st.subheader("Selected Models")
        for model_key in selected_models:
            config = MODEL_CONFIGS[model_key]
            st.write(f"- **{model_key}**: `{config['model_name']}`")
        
        st.subheader("Command Preview")
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
            st.info(f"🐍 Using global conda environment: **{conda_env}**")
        else:
            st.caption("💡 Set conda environment in Settings page for consistent usage")
        
        if st.button("Download Models", type="primary", use_container_width=True):
            st.info(f"Starting model download with conda environment '{conda_env}'...")
            
            # Get direct python path from conda environment
            python_path = _get_conda_python_path(conda_env)
            if python_path:
                # Build command with direct python path and selected models
                model_args = " ".join(MODEL_CONFIGS[key]["model_key"] for key in selected_models)
                direct_cmd = f"{python_path} -u download_models.py --models {model_args}"
                if save_dir != "./models":
                    direct_cmd += f" --save-dir {shlex.quote(save_dir)}"
                
                st.info(f"Using Python: `{python_path}`")
                st.info(f"Executing: `{direct_cmd}`")
                
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
                    
                    # Display process ID
                    st.info(f"🔧 **Process ID (PID): {process.pid}** - You can manually stop it using: `kill {process.pid}`")
                    
                    stdout_lines = []
                    stderr_lines = []

                    output_section = st.container()
                    output_section.subheader("📋 Download Output (Live)")
                    output_section.caption("Shows real-time stdout/stderr from the download process.")
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
                                stderr_container.error("**Errors/Warnings:**")
                                stderr_container.code("".join(stderr_lines), language="bash")
                            last_ui_update = now

                    for name, rest in buffers.items():
                        if rest:
                            target = stdout_lines if name == "stdout" else stderr_lines
                            target.append(rest + "\n")

                    if stdout_lines:
                        stdout_container.code("".join(stdout_lines), language="bash")
                    if stderr_lines:
                        stderr_container.error("**Errors/Warnings:**")
                        stderr_container.code("".join(stderr_lines), language="bash")
                    
                    return_code = process.wait()
                    
                    # Show execution result
                    if return_code == 0:
                        st.success(f"✅ Model download completed successfully (exit code: {return_code})")
                        st.info(f"📁 **Models saved to:** `{os.path.join(PROJECT_ROOT, save_dir)}`")
                        st.caption("You can now use these models in your training scripts.")
                    else:
                        st.error(f"❌ Model download failed with exit code: {return_code}")
                        
                except Exception as e:
                    st.error(f"Error executing command: {e}")
            else:
                st.error(f"❌ Could not find Python executable for conda environment '{conda_env}'")
                st.info("Please check if the environment exists and try setting it in Settings page.")
