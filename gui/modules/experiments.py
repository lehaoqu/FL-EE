"""Experiment Overview page — tabular summary of all training/eval artefacts."""

import os
from typing import List

import pandas as pd
import streamlit as st

from modules._utils import (
    PROJECT_ROOT,
    discover_result_roots,
    scan_experiments,
)

_BADGE = (
    "background:{bg};color:{fg};padding:2px 10px;border-radius:4px;"
    "font-size:11px;font-weight:700;letter-spacing:0.6px;text-transform:uppercase;"
)


def _section_label(text: str, color: str = "#ff8c00") -> None:
    bg = color + "1a"
    style = _BADGE.format(bg=bg, fg=color)
    st.markdown(f"<span style='{style}'>{text}</span>", unsafe_allow_html=True)


def _jump_to(page: str) -> None:
    """Request a cross-page navigation on next rerun."""
    st.session_state["_target_page"] = page
    st.rerun()


def show() -> None:
    st.markdown(
        "<div style='font-size:2.4rem;font-weight:700;line-height:1.2;margin-bottom:2px'>📊 实验总览</div>"
        "<p style='color:#6b7280;font-size:14px;margin-top:0'>扫描所有结果根目录，汇总实验状态，可筛选、排序、一键跳转。</p>",
        unsafe_allow_html=True,
    )

    # ---- Choose which result roots to scan ---------------------------------
    all_candidates = discover_result_roots()
    if not all_candidates:
        st.info("项目下暂未发现任何结果目录（如 front-exps / EXPS* 等）。")
        return

    defaults = [r for r in ["front-exps", "EXPS", "EXPS2"] if r in all_candidates]
    if not defaults:
        defaults = all_candidates[:1]

    # Only write to session_state once, before instantiating the widget
    st.session_state.setdefault("exp_overview_roots", defaults)

    with st.expander("扫描范围", expanded=False):
        roots = st.multiselect(
            "结果根目录（可多选）",
            options=all_candidates,
            default=None,  # let the widget read from session_state via key
            key="exp_overview_roots",
            help="勾选需要扫描的结果目录，所有结果将聚合展示。",
        )

    if not roots:
        st.info("请至少选择一个结果根目录。")
        return

    experiments = scan_experiments(roots)
    if not experiments:
        st.warning("所选目录下暂未发现任何 .pth 或 *_eval.json 文件。")
        return

    df = pd.DataFrame(experiments)

    # ---- Top-level metrics --------------------------------------------------
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实验总数", len(df))
    c2.metric("已训练", int(df["has_model"].sum()))
    c3.metric("已评估", int(df["has_eval"].sum()))
    c4.metric("扫描根目录", len(roots))

    # ---- Filters ------------------------------------------------------------
    with st.expander("筛选", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            algos = sorted(df["policy"].unique())
            sel_algos = st.multiselect("算法", algos, default=algos)
        with fc2:
            datasets = sorted(df["dataset"].unique())
            sel_datasets = st.multiselect("数据集", datasets, default=datasets)
        with fc3:
            models = sorted(df["model"].unique())
            sel_models = st.multiselect("模型", models, default=models)
        with fc4:
            status_opts = ["已训练(有.pth)", "已评估(有_eval.json)", "未评估"]
            sel_status = st.multiselect("状态", status_opts, default=status_opts)

        keyword = st.text_input(
            "关键词过滤（匹配路径 / 文件名）",
            value="",
            placeholder="例如：noniid1000、boosted、lora",
        )

    mask = (
        df["policy"].isin(sel_algos)
        & df["dataset"].isin(sel_datasets)
        & df["model"].isin(sel_models)
    )

    status_mask = pd.Series(False, index=df.index)
    if "已训练(有.pth)" in sel_status:
        status_mask |= df["has_model"]
    if "已评估(有_eval.json)" in sel_status:
        status_mask |= df["has_eval"]
    if "未评估" in sel_status:
        status_mask |= ~df["has_eval"]

    if keyword:
        kw = keyword.lower()
        kw_mask = (
            df["rel_dir"].str.lower().str.contains(kw, na=False)
            | df["stem"].str.lower().str.contains(kw, na=False)
            | df["root"].str.lower().str.contains(kw, na=False)
        )
    else:
        kw_mask = pd.Series(True, index=df.index)

    df_view = df[mask & status_mask & kw_mask].copy()
    df_view = df_view.sort_values("last_modified_ts", ascending=False)

    st.caption(f"筛选后：{len(df_view)} / {len(df)} 条实验记录")

    # ---- Table --------------------------------------------------------------
    display_cols = [
        "policy", "dataset", "model", "mode", "ft",
        "has_model", "has_eval", "peak_acc",
        "last_modified", "root", "rel_dir", "stem",
    ]
    display_df = df_view[display_cols].rename(columns={
        "policy": "算法",
        "dataset": "数据集",
        "model": "模型",
        "mode": "模式",
        "ft": "微调",
        "has_model": "有模型",
        "has_eval": "有评估",
        "peak_acc": "Peak Acc",
        "last_modified": "最后修改",
        "root": "根目录",
        "rel_dir": "子路径",
        "stem": "实验标识",
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(600, 80 + 32 * min(len(display_df), 18)),
        column_config={
            "有模型": st.column_config.CheckboxColumn(),
            "有评估": st.column_config.CheckboxColumn(),
            "Peak Acc": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ---- Export -------------------------------------------------------------
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 导出当前视图为 CSV",
        data=csv_bytes,
        file_name="experiments_overview.csv",
        mime="text/csv",
    )

    # ---- Quick jump ---------------------------------------------------------
    _section_label("快速跳转", "#1a73e8")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    jc1, jc2, jc3, jc4 = st.columns(4)
    with jc1:
        if st.button("🚀 训练新实验", use_container_width=True):
            _jump_to("训练与监控")
    with jc2:
        if st.button("🔬 模型评估", use_container_width=True):
            _jump_to("模型评估")
    with jc3:
        if st.button("🗄️ 数据集生成", use_container_width=True):
            _jump_to("数据集生成")
    with jc4:
        if st.button("📈 数据分布", use_container_width=True):
            _jump_to("数据分布")

    # ---- Per-experiment drilldown ------------------------------------------
    if not df_view.empty:
        with st.expander("实验明细（选择一行查看）", expanded=False):
            options = [
                f"[{r['root']}] {r['rel_dir']} :: {r['stem']}"
                for _, r in df_view.iterrows()
            ]
            choice = st.selectbox("实验", options=options)
            if choice:
                idx = options.index(choice)
                r = df_view.iloc[idx]
                full_dir = os.path.join(PROJECT_ROOT, str(r["root"]), str(r["rel_dir"]))
                st.write(f"**完整路径**：`{full_dir}`")
                st.write(f"**算法 / 数据集 / 模型 / 模式 / 微调**：`{r['policy']}` / `{r['dataset']}` / `{r['model']}` / `{r['mode']}` / `{r['ft']}`")
                if r["slim"]:
                    st.write(f"**slim**：`{r['slim']}`")
                if r["exits"]:
                    st.write(f"**exits**：`{r['exits']}`")
                st.write(f"**有模型**：{'✅' if r['has_model'] else '❌'}  "
                         f"**有评估**：{'✅' if r['has_eval'] else '❌'}")
                if r["has_eval"] and not pd.isna(r["peak_acc"]):
                    st.write(f"**Peak Acc**：`{r['peak_acc']:.4f}`")
                if r["has_eval"]:
                    st.caption(f"eval.json：`{r['eval_json_path']}`")
