"""Non-IID distribution visualisation for generated datasets and Dirichlet preview."""

from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from modules._utils import scan_datasets


def show() -> None:
    st.markdown(
        "<div style='font-size:2.4rem;font-weight:700;line-height:1.2;margin-bottom:2px'>📈 数据分布可视化</div>"
        "<p style='color:#6b7280;font-size:14px;margin-top:0'>"
        "查看各客户端在联邦学习场景下的类别分布，或模拟 Dirichlet 划分。</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tab_existing, tab_sim = st.tabs(["已生成数据集", "Dirichlet 模拟"])

    with tab_existing:
        _show_existing_datasets()

    with tab_sim:
        _show_dirichlet_simulator()


# ---------------------------------------------------------------------------
# Existing datasets
# ---------------------------------------------------------------------------

def _show_existing_datasets() -> None:
    datasets = scan_datasets()
    if not datasets:
        st.info("未在 `dataset/` 下发现任何已生成数据集（需包含 config.json）。")
        return

    name_to_dict: Dict[str, Dict] = {d["name"]: d for d in datasets}
    selected_name = st.selectbox(
        "选择数据集",
        options=list(name_to_dict.keys()),
        help="只列出包含 config.json 的子目录。",
    )
    d = name_to_dict[selected_name]

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("客户端", d["num_clients"])
    cc2.metric("类别", d["num_classes"])
    cc3.metric("样本总数", f"{d['total_samples']:,}")
    cc4.metric("Non-IID", "是" if d["non_iid"] else "否")
    cc5.metric("划分方式", d["partition"] or "-")

    matrix: np.ndarray = d["matrix"]
    if matrix.size == 0 or matrix.sum() == 0:
        st.warning("config.json 中未找到 `Size of samples for labels in clients`，无法绘制分布。")
        return

    # Controls
    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        max_clients = st.slider(
            "显示客户端数（前 N 个）",
            min_value=min(5, d["num_clients"]),
            max_value=int(d["num_clients"]),
            value=int(min(30, d["num_clients"])),
        )
    with ctl2:
        viz_type = st.radio(
            "可视化类型",
            ["堆叠柱状图", "类别分布热图", "每客户端类别数", "每客户端样本量"],
            horizontal=True,
        )

    m = matrix[:max_clients]

    stats_matrix = matrix[: min(100, matrix.shape[0])]

    if viz_type == "堆叠柱状图":
        _plot_stacked_bar(m)
    elif viz_type == "类别分布热图":
        _plot_heatmap(m, num_classes=d["num_classes"])
    elif viz_type == "每客户端类别数":
        _plot_classes_per_client(stats_matrix, num_classes=d["num_classes"])
    elif viz_type == "每客户端样本量":
        _plot_samples_per_client(stats_matrix)


def _plot_stacked_bar(m: np.ndarray) -> None:
    rows = []
    num_clients, num_classes = m.shape
    for ci in range(num_clients):
        for cls in range(num_classes):
            cnt = int(m[ci, cls])
            if cnt > 0:
                rows.append({"client": ci, "class": cls, "count": cnt})
    if not rows:
        st.warning("所选范围内没有样本。")
        return
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("client:O", title="Client ID", sort=None),
            y=alt.Y("count:Q", title="Sample count"),
            color=alt.Color(
                "class:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="Class idx"),
            ),
            tooltip=["client", "class", "count"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("每一条柱代表一个客户端，颜色段表示不同类别的样本量。")


def _plot_heatmap(m: np.ndarray, num_classes: int) -> None:
    import matplotlib.pyplot as plt

    num_clients = m.shape[0]
    fig, ax = plt.subplots(figsize=(12, max(4.0, num_clients * 0.18)))
    im = ax.imshow(m, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Class index")
    ax.set_ylabel("Client")
    ax.set_title(f"Per-client class distribution (first {num_clients} clients, {num_classes} classes)")
    fig.colorbar(im, ax=ax, label="Sample count")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def _plot_classes_per_client(full_matrix: np.ndarray, num_classes: int) -> None:
    counts = (full_matrix > 0).sum(axis=1)
    df = pd.DataFrame({"client": np.arange(len(counts)), "num_classes": counts})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("num_classes:Q", bin=alt.Bin(maxbins=30), title=f"Number of classes per client (total={num_classes})"),
            y=alt.Y("count()", title="Clients"),
            tooltip=["count()"],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"按 {len(counts)} 个客户端统计：均值 {counts.mean():.2f} / {num_classes}，标准差 {counts.std():.2f}，"
        f"最小 {int(counts.min())}，最大 {int(counts.max())}。"
    )


def _plot_samples_per_client(full_matrix: np.ndarray) -> None:
    totals = full_matrix.sum(axis=1)
    df = pd.DataFrame({"client": np.arange(len(totals)), "total": totals})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("total:Q", bin=alt.Bin(maxbins=30), title="Total samples per client"),
            y=alt.Y("count()", title="Clients"),
            tooltip=["count()"],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"按 {len(totals)} 个客户端统计：均值 {totals.mean():.0f}，标准差 {totals.std():.0f}，"
        f"最小 {int(totals.min())}，最大 {int(totals.max())}。"
    )


# ---------------------------------------------------------------------------
# Dirichlet simulator
# ---------------------------------------------------------------------------

def _show_dirichlet_simulator() -> None:
    st.caption("在生成数据集之前预览 Dirichlet(α) 会产生怎样的客户端类别分布。")

    c1, c2, c3 = st.columns(3)
    with c1:
        num_clients = st.number_input("客户端数", min_value=2, max_value=1000, value=100, step=10)
    with c2:
        num_classes = st.number_input("类别数", min_value=2, max_value=1000, value=100, step=10)
    with c3:
        alpha = st.number_input("Alpha（Dirichlet 浓度）", min_value=0.001, max_value=1000.0, value=1.0, step=0.1, format="%.3f")

    c4, c5 = st.columns(2)
    with c4:
        total_per_class = st.number_input("每类样本总数", min_value=10, max_value=100000, value=500, step=50)
    with c5:
        seed = st.number_input("随机种子", min_value=0, max_value=99999, value=42, step=1)

    rng = np.random.default_rng(int(seed))
    proportions = rng.dirichlet(np.full(int(num_clients), float(alpha)), size=int(num_classes))
    matrix = (proportions.T * float(total_per_class)).round().astype(int)  # (clients, classes)

    max_display = int(min(40, num_clients))
    st.write(f"展示前 {max_display} 个客户端的类别分布（α={alpha}）")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.0))
    im = ax.imshow(matrix[:max_display].T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Client")
    ax.set_ylabel("Class index")
    ax.set_title(f"Simulated Dirichlet(α={alpha}) — first {max_display} clients")
    fig.colorbar(im, ax=ax, label="Sample count")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    counts = (matrix > 0).sum(axis=1)
    totals = matrix.sum(axis=1)
    st.caption(
        f"平均类别数/客户端：{counts.mean():.2f} / {num_classes}（std {counts.std():.2f}）｜ "
        f"平均样本/客户端：{totals.mean():.0f}（std {totals.std():.0f}）"
    )

    with st.expander("提示"):
        st.markdown(
            "- α **越小**：每个客户端拥有的类别越少，异质性越强（更 Non-IID）。\n"
            "- α **越大**：分布趋向均匀，接近 IID。\n"
            "- 模拟与真实生成脚本可能略有差异（实际脚本可能带平衡逻辑）。"
        )
