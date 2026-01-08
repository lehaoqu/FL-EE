import json
import os
import subprocess

import streamlit as st


def _discover_conda_envs():
    """Return sorted list of available conda environments detected on the host."""
    try:
        result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout or "{}")
        env_paths = data.get("envs", [])
        env_names = [os.path.basename(path) for path in env_paths if isinstance(path, str) and path]
        return sorted(dict.fromkeys(env_names))
    except Exception:
        return []


def show():
    """显示 Settings 页面"""
    st.header("Settings")
    st.write("Application settings")
    
    # Global Conda Environment Setting
    st.subheader("🐍 Python Environment")
    env_options = _discover_conda_envs()
    
    # Add System Python option
    all_options = ["System Python (python3)"] + env_options
    
    if env_options:
        default_index = all_options.index("fl-ee") if "fl-ee" in all_options else 0
        selected_env = st.selectbox(
            "Python Environment",
            all_options,
            index=default_index,
            help="Select 'System Python' to use system python3, or choose a conda environment",
            key="global_conda_env"
        )
        st.session_state["conda_env"] = selected_env
        if selected_env == "System Python (python3)":
            st.success(f"✅ Using system Python: **python3**")
        else:
            st.success(f"✅ Using conda environment: **{selected_env}**")
    else:
        st.warning("⚠️ Conda environment detection failed")
        fallback_options = ["System Python (python3)", "fl-ee"]
        selected_env = st.selectbox(
            "Python Environment",
            fallback_options,
            help="Select 'System Python' or enter custom environment name below",
            key="fallback_env_select"
        )
        if selected_env != "System Python (python3)":
            manual_env = st.text_input(
                "Or enter custom environment name", 
                value=selected_env,
                help="Manually specify the conda environment name",
                key="manual_conda_env"
            )
            st.session_state["conda_env"] = manual_env
            st.info(f"🔧 Using manually specified environment: **{manual_env}**")
        else:
            st.session_state["conda_env"] = selected_env
            st.success(f"✅ Using system Python: **python3**")
    
    st.caption("This setting affects all training and dataset generation operations.")
    
    st.divider()
    
    # Other settings
    st.subheader("⚙️ General Settings")
    st.checkbox("Enable verbose logging", value=False)
    st.selectbox("Default dataset", ["cifar100", "imagenet", "glue", "speechcmd", "svhn"])
