import streamlit as st


def show():
    """显示 Evaluate 页面"""
    st.header("Evaluate")
    st.write("Run evaluations and view results.")
    dataset = st.selectbox("Dataset for evaluation", ["cifar100", "imagenet", "glue", "speechcmd", "svhn"], key="eval_dataset")
    exp_dir = st.text_input("Experiment directory", "EXPS/test/")
    if st.button("Run evaluation"):
        st.info(f"Evaluating {exp_dir} on {dataset}... (this is a placeholder)")
