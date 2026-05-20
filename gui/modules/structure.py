from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st


@dataclass(frozen=True)
class KeyPath:
    label: str
    rel_path: str
    desc: str


# 关键目录/文件（偏“怎么用/去哪改”视角）
KEY_PATHS: list[KeyPath] = [
    KeyPath("训练入口", "main.py", "主入口：训练/联邦仿真流程"),
    KeyPath("评估入口", "eval.py", "评估入口"),
    KeyPath("Slimmable评估", "eval_slimmable.py", "Slimmable/可伸缩模型评估"),
    KeyPath("全局配置", "global.py", "项目级全局配置/变量"),
    KeyPath("数据集与划分", "dataset", "数据集定义、划分工具与生成后的划分结果"),
    KeyPath("算法实现", "trainer/alg", "各联邦算法实现（EEF L / DarkFL* / ScaleFL 等）"),
    KeyPath("训练框架", "trainer", "Client/Server 基类、策略、生成器等"),
    KeyPath("通用参数", "utils/options.py", "通用命令行参数与默认值"),
    KeyPath("运行脚本", "script", "常用实验运行脚本（bash）"),
    KeyPath("GUI入口", "gui/front.py", "Streamlit GUI 入口"),
    KeyPath("模型缓存", "models", "预训练/下载模型缓存"),
    KeyPath("实验输出", "exps", "实验配置与输出（通常很大）"),
    KeyPath("可视化输出", "imgs", "绘图输出"),
]


DEFAULT_IGNORES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "wandb",
}


def _iter_children(path: Path) -> list[Path]:
    try:
        children = list(path.iterdir())
    except PermissionError:
        return []

    # 目录优先，其次文件；同类按名称排序
    children.sort(key=lambda p: (p.is_file(), p.name.lower()))
    return children


def _should_ignore(p: Path, ignores: set[str]) -> bool:
    return p.name in ignores


def build_tree(
    root: Path,
    *,
    max_depth: int = 3,
    show_files: bool = True,
    max_entries_per_dir: int = 200,
    ignores: set[str] | None = None,
) -> str:
    """生成一个适合在 Streamlit 展示的 ASCII tree。"""

    if ignores is None:
        ignores = set(DEFAULT_IGNORES)

    root = root.resolve()
    lines: list[str] = [str(root.name) + "/"]

    def walk(dir_path: Path, depth: int, prefix: str) -> None:
        if depth >= max_depth:
            return

        children = [p for p in _iter_children(dir_path) if not _should_ignore(p, ignores)]

        if not show_files:
            children = [p for p in children if p.is_dir()]

        truncated = False
        if len(children) > max_entries_per_dir:
            children = children[:max_entries_per_dir]
            truncated = True

        for idx, child in enumerate(children):
            is_last = idx == (len(children) - 1) and not truncated
            connector = "└─ " if is_last else "├─ "
            display_name = child.name + ("/" if child.is_dir() else "")
            lines.append(prefix + connector + display_name)

            if child.is_dir():
                extension = "   " if is_last else "│  "
                walk(child, depth + 1, prefix + extension)

        if truncated:
            lines.append(prefix + "└─ ... (truncated)")

    walk(root, 0, "")
    return "\n".join(lines)


def _repo_root() -> Path:
    # gui/modules/structure.py -> gui/modules -> gui -> repo root
    return Path(__file__).resolve().parents[2]


def _existing_key_paths(root: Path) -> list[KeyPath]:
    existing: list[KeyPath] = []
    for kp in KEY_PATHS:
        if (root / kp.rel_path).exists():
            existing.append(kp)
    return existing


def show() -> None:
    st.markdown(
        "<div style='font-size:2.4rem;font-weight:700;line-height:1.2;margin-bottom:2px'>🗂️ 项目结构</div>"
        "<p style='color:#6b7280;font-size:14px;margin-top:0'>"
        "浏览 FL-EE 的关键目录与文件。</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    root = _repo_root()

    with st.expander("关键目录说明", expanded=True):
        for kp in _existing_key_paths(root):
            st.markdown(f"- **{kp.label}**：`{kp.rel_path}` — {kp.desc}")

    st.subheader("目录树")

    selectable: list[tuple[str, Path]] = [("仓库根目录", root)]
    for kp in _existing_key_paths(root):
        selectable.append((f"{kp.label}：{kp.rel_path}", root / kp.rel_path))

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_name = st.selectbox("选择要展示的目录/文件", [name for name, _ in selectable], index=0)
    with col2:
        max_depth = st.slider("深度", min_value=1, max_value=8, value=4, step=1)
    with col3:
        show_files = st.checkbox("显示文件", value=True)

    selected_path = dict(selectable)[selected_name]

    # 如果选中的是文件，则展示其所在目录并高亮提示
    if selected_path.is_file():
        st.info(f"你选择的是文件：`{selected_path.relative_to(root)}`。下面展示其所在目录树。")
        tree_root = selected_path.parent
    else:
        tree_root = selected_path

    tree_text = build_tree(
        tree_root,
        max_depth=max_depth,
        show_files=show_files,
        max_entries_per_dir=200,
    )

    st.code(tree_text, language="text")
    print(tree_text)

    with st.expander("提示", expanded=False):
        st.markdown(
            "- 默认隐藏：`.git`、`.venv`、`__pycache__`、`wandb` 等目录（避免噪音）。\n"
            "- 如果要看实验输出（`exps/` 等）更深层级，可提高深度，但目录可能很大。"
        )
