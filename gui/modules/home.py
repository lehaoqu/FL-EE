"""Dashboard home page — stats, recent experiments, quick actions."""

import altair as alt
import pandas as pd
import streamlit as st

from modules._utils import (
    discover_result_roots,
    scan_datasets,
    scan_experiments,
)

_BADGE = (
    "background:{bg};color:{fg};padding:4px 12px;border-radius:4px;"
    "font-size:20px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;"
)


def _section_label(text: str, color: str = "#ff8c00") -> None:
    bg = color + "1a"
    style = _BADGE.format(bg=bg, fg=color)
    st.markdown(f"<span style='{style}'>{text}</span>", unsafe_allow_html=True)


def _jump_to(page: str) -> None:
    st.session_state["_target_page"] = page
    st.rerun()


def show() -> None:
    # ---- Hero area ----------------------------------------------------------
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            "<div style='margin-bottom:2px;font-size:2.4rem;font-weight:700;line-height:1.2'>FL-EE 控制台</div>"
            "<p style='color:#6b7280;font-size:14px;margin-top:0'>Federated Learning with Early Exit — 实验管理与可视化平台</p>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        conda_env = st.session_state.get("conda_env", "—")
        env_display = conda_env if len(conda_env) <= 18 else conda_env[:16] + "…"
        st.markdown(
            f"<div style='text-align:right;color:#9ca3af;font-size:20px;padding-top:12px'>"
            f"🐍 <b style='color:#374151'>{env_display}</b></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ---- Top stats ----------------------------------------------------------
    roots = discover_result_roots()
    print(roots)
    experiments = scan_experiments(roots) if roots else []
    datasets = scan_datasets()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("实验总数", len(experiments))
    c2.metric("已训练", sum(1 for e in experiments if e["has_model"]))
    c3.metric("已评估", sum(1 for e in experiments if e["has_eval"]))
    c4.metric("数据集", len(datasets))
    c5.metric("扫描根目录", len(roots))

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ---- Quick actions ------------------------------------------------------
    _section_label("快速入口")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    with qa1:
        if st.button("📊 实验总览", use_container_width=True):
            _jump_to("实验总览")
    with qa2:
        if st.button("📈 数据分布", use_container_width=True):
            _jump_to("数据分布")
    with qa3:
        if st.button("🚀 开始训练", use_container_width=True):
            _jump_to("训练与监控")
    with qa4:
        if st.button("🔬 运行评估", use_container_width=True):
            _jump_to("模型评估")
    with qa5:
        if st.button("🗄️ 生成数据集", use_container_width=True):
            _jump_to("数据集生成")

    st.divider()

    # ---- Row 1: Recent experiments + Device resources ---
    if experiments:
        df_all = pd.DataFrame(experiments)

        left_col, right_col = st.columns([3, 2], gap="medium")
        TABLE_HEIGHT = 410

        with left_col:
            _section_label("最近实验", "#1a73e8")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            df = df_all.sort_values("last_modified_ts", ascending=False).head(10)
            show_df = df[[
                "policy", "dataset", "model", "mode",
                "has_model", "has_eval", "peak_acc", "last_modified",
            ]].rename(columns={
                "policy": "算法",
                "dataset": "数据集",
                "model": "模型",
                "mode": "模式",
                "has_model": "已训练",
                "has_eval": "已评估",
                "peak_acc": "Peak Acc",
                "last_modified": "最后修改",
            })
            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True,
                height=TABLE_HEIGHT,
                column_config={
                    "已训练": st.column_config.CheckboxColumn(),
                    "已评估": st.column_config.CheckboxColumn(),
                    "Peak Acc": st.column_config.NumberColumn(format="%.4f"),
                },
            )

        with right_col:
            _section_label("设备资源分布", "#f59e0b")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # Store raw weights — only normalize at display time, never overwrite user input.
            st.session_state.setdefault("device_weights", {"CPU": 0.2, "GPU": 0.4, "TPU": 0.3, "边缘设备": 0.1})

            weights = st.session_state.device_weights
            total = sum(weights.values()) or 1.0
            ratios = {k: v / total for k, v in weights.items()}
            # Persist normalized ratios for downstream consumers (training page, etc.).
            st.session_state.device_ratios = ratios

            # Keep a stable sort so the pie arcs and text labels align 1:1.
            device_data = pd.DataFrame([
                {"设备类型": k, "比例": v, "百分比": f"{v:.1%}", "_ord": i}
                for i, (k, v) in enumerate(ratios.items())
            ])

            theta = alt.Theta("比例:Q", stack=True)
            order = alt.Order("_ord:Q")
            color = alt.Color(
                "设备类型:N",
                scale=alt.Scale(scheme="set2"),
                legend=alt.Legend(
                    orient="bottom-right",
                    direction="vertical",
                    labelFontSize=12,
                    symbolSize=120,
                    offset=10,
                    padding=4,
                    title=None,
                ),
            )

            pie = (
                alt.Chart(device_data)
                .mark_arc(innerRadius=0, outerRadius=115, cornerRadius=3)
                .encode(
                    theta=theta,
                    order=order,
                    color=color,
                    tooltip=["设备类型", alt.Tooltip("比例:Q", format=".1%")],
                )
            )
            labels = (
                alt.Chart(device_data)
                .mark_text(
                    radius=75,
                    fontSize=16,
                    fontWeight=700,
                    color="#ffffff",
                    stroke="#000000",
                    strokeWidth=0.4,
                )
                .encode(
                    theta=theta,
                    order=order,
                    text=alt.Text("百分比:N"),
                )
            )
            chart = (pie + labels).properties(height=320).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

            # Edit device ratios — values are only normalized when rendering, not while typing
            with st.expander("⚙️ 编辑设备权重", expanded=False):
                st.markdown(
                    "<div style='font-size:13px;color:#6b7280;margin-bottom:8px'>"
                    "输入各设备的权重值（可任意数值，饼图会按比例显示）</div>",
                    unsafe_allow_html=True,
                )

                with st.form("device_weights_form", border=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        gpu_val = st.number_input("GPU", min_value=0.0, value=float(weights.get("GPU", 0.4)), step=0.1, format="%.2f", key="_inp_gpu")
                        cpu_val = st.number_input("CPU", min_value=0.0, value=float(weights.get("CPU", 0.2)), step=0.1, format="%.2f", key="_inp_cpu")
                    with c2:
                        tpu_val = st.number_input("TPU", min_value=0.0, value=float(weights.get("TPU", 0.3)), step=0.1, format="%.2f", key="_inp_tpu")
                        edge_val = st.number_input("边缘设备", min_value=0.0, value=float(weights.get("边缘设备", 0.1)), step=0.1, format="%.2f", key="_inp_edge")

                    submitted = st.form_submit_button("应用", type="primary", use_container_width=True)
                    if submitted:
                        if gpu_val + tpu_val + cpu_val + edge_val <= 0:
                            st.warning("至少需要一个非零值")
                        else:
                            st.session_state.device_weights = {
                                "CPU": cpu_val, "GPU": gpu_val, "TPU": tpu_val, "边缘设备": edge_val,
                            }
                            st.rerun()

    # ---- Row 2: Distribution overview (full width) ---
    if experiments:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _section_label("分布概览", "#34a853")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        dist_left, dist_right = st.columns(2, gap="medium")
        CHART_HEIGHT = 220

        with dist_left:
            algo_counts = (
                df_all.groupby("policy").size().reset_index(name="count")
                .sort_values("count", ascending=False).head(8)
            )
            st.markdown(
                "<div style='font-size:16px;font-weight:600;color:#374151;margin:0 0 2px 0'>按算法统计 (Top 8)</div>",
                unsafe_allow_html=True,
            )
            chart = (
                alt.Chart(algo_counts)
                .mark_bar(cornerRadiusBottomRight=4, cornerRadiusTopRight=4)
                .encode(
                    y=alt.Y("policy:N", title=None, sort="-x"),
                    x=alt.X("count:Q", title="实验数"),
                    color=alt.Color("policy:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                    tooltip=["policy", "count"],
                )
                .properties(height=CHART_HEIGHT)
            )
            st.altair_chart(chart, use_container_width=True)

        with dist_right:
            ds_counts = (
                df_all.groupby("dataset").size().reset_index(name="count")
                .sort_values("count", ascending=False).head(8)
            )
            st.markdown(
                "<div style='font-size:16px;font-weight:600;color:#374151;margin:0 0 2px 0'>按数据集统计 (Top 8)</div>",
                unsafe_allow_html=True,
            )
            chart2 = (
                alt.Chart(ds_counts)
                .mark_bar(cornerRadiusBottomRight=4, cornerRadiusTopRight=4)
                .encode(
                    y=alt.Y("dataset:N", title=None, sort="-x"),
                    x=alt.X("count:Q", title="实验数"),
                    color=alt.Color("dataset:N", legend=None, scale=alt.Scale(scheme="tableau20")),
                    tooltip=["dataset", "count"],
                )
                .properties(height=CHART_HEIGHT)
            )
            st.altair_chart(chart2, use_container_width=True)

    else:
        st.info("暂未发现实验记录。完成一次训练后即可自动汇总。")

    # ---- Datasets summary ---------------------------------------------------
    if datasets:
        st.divider()
        _section_label("已生成数据集", "#7c3aed")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        ds_rows = [
            {
                "名称": d["name"],
                "客户端": d["num_clients"],
                "类别": d["num_classes"],
                "样本总数": f"{d['total_samples']:,}",
                "Non-IID": "✅" if d["non_iid"] else "—",
                "划分": d["partition"] or "-",
            }
            for d in datasets
        ]
        st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True)
