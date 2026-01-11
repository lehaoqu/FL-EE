import json
import os
import subprocess

import streamlit as st


def _discover_conda_envs():
    """返回主机上检测到的可用 conda 环境列表（排序）。"""
    try:
        result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout or "{}")
        env_paths = data.get("envs", [])
        env_names = [os.path.basename(path) for path in env_paths if isinstance(path, str) and path]
        return sorted(dict.fromkeys(env_names))
    except Exception:
        return []


def show():
    """显示设置页面"""
    st.header("设置")
    st.write("应用设置")
    
    # 全局 Conda 环境设置
    st.subheader("🐍 Python 环境")
    env_options = _discover_conda_envs()
    
    # 添加 System Python 选项
    all_options = ["System Python (python)"] + env_options
    
    if env_options:
        default_index = all_options.index("fl-ee") if "fl-ee" in all_options else 0
        selected_env = st.selectbox(
            "Python 环境",
            all_options,
            index=default_index,
            help="选择 System Python 使用系统 Python，或选择一个 conda 环境",
            key="global_conda_env"
        )
        st.session_state["conda_env"] = selected_env
        if selected_env == "System Python (python)":
            st.success(f"✅ 使用系统 Python：**python**")
        else:
            st.success(f"✅ 使用 Conda 环境：**{selected_env}**")
    else:
        st.warning("⚠️ 未能检测到 Conda 环境")
        fallback_options = ["System Python (python)", "fl-ee"]
        selected_env = st.selectbox(
            "Python 环境",
            fallback_options,
            help="选择 System Python 或在下方输入自定义环境名",
            key="fallback_env_select"
        )
        if selected_env != "System Python (python)":
            manual_env = st.text_input(
                "或手动输入环境名", 
                value=selected_env,
                help="手动指定 conda 环境名称",
                key="manual_conda_env"
            )
            st.session_state["conda_env"] = manual_env
            st.info(f"🔧 使用手动指定的环境：**{manual_env}**")
        else:
            st.session_state["conda_env"] = selected_env
            st.success(f"✅ 使用系统 Python：**python**")
    
    st.caption("此设置会影响所有训练和数据集生成操作。")
    
    st.divider()
    
    # Hugging Face endpoint（离线镜像/代理）
    st.subheader("🌐 Hugging Face 地址")
    default_hf = st.session_state.get("hf_endpoint", os.environ.get("HF_ENDPOINT", ""))
    hf_endpoint = st.text_input(
        "HF_ENDPOINT",
        value=default_hf,
        help="设置镜像或代理，例如 https://hf-mirror.com，留空使用默认。"
    )
    if hf_endpoint != st.session_state.get("hf_endpoint"):
        st.session_state["hf_endpoint"] = hf_endpoint
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            st.success(f"✅ HF_ENDPOINT 已设置为：{hf_endpoint}")
        else:
            os.environ.pop("HF_ENDPOINT", None)
            st.info("HF_ENDPOINT 已清除；使用默认 Hugging Face 地址")

    st.divider()

    # 其他设置
    st.subheader("⚙️ 常规设置")
    st.checkbox("启用详细日志", value=False)
    st.selectbox("默认数据集", ["cifar100", "imagenet", "glue", "speechcmd", "svhn"])
