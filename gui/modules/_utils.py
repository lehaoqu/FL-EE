"""Shared scan + UI utilities for the GUI pages."""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared UI helpers (used by every page for a consistent look)
# ---------------------------------------------------------------------------

_SECTION_BADGE_CSS = (
    "background:{bg};color:{fg};padding:4px 12px;border-radius:4px;"
    "font-size:20px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;"
    "display:inline-block;margin-bottom:6px;"
)


def section_label(text: str, color: str = "#ff8c00") -> None:
    """Render a colored badge used as a section header."""
    bg = color + "1a"  # ~10% opacity
    style = _SECTION_BADGE_CSS.format(bg=bg, fg=color)
    st.markdown(f"<span style='{style}'>{text}</span>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Unified page header: large title + optional subtitle."""
    parts = [
        f"<div style='font-size:2.4rem;font-weight:700;line-height:1.2;margin-bottom:2px'>{title}</div>"
    ]
    if subtitle:
        parts.append(
            f"<p style='color:#6b7280;font-size:14px;margin-top:0'>{subtitle}</p>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def info_card(icon: str, title: str, description: str, color: str = "#1a73e8") -> None:
    """Render a small tinted info card with a left accent stripe."""
    st.markdown(
        f"""<div style='background:{color}10;border-left:3px solid {color};
                       padding:10px 14px;border-radius:6px;margin:4px 0 10px 0;'>
            <div style='font-weight:600;color:{color};font-size:14px'>{icon} {title}</div>
            <div style='color:#374151;font-size:13px;margin-top:3px;line-height:1.5'>{description}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Experiment scanning
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> Tuple[str, str, str, str]:
    """Parse (policy, dataset, model, mode) from an experiment filename stem.

    Example: `eefl_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted`.
    Mirrors the logic in modules/evaluate.py so results stay consistent.
    """
    tokens = stem.split("_")
    policy = tokens[0] if tokens else ""

    model = ""
    model_idx: Optional[int] = None
    for i, tok in enumerate(tokens):
        if tok in {"vit", "bert"}:
            model = tok
            model_idx = i
            break

    dataset = ""
    if model_idx is not None and model_idx >= 2:
        dataset = "_".join(tokens[1:model_idx])
    elif len(tokens) >= 2:
        dataset = tokens[1]

    mode = ""
    for cand in ("boosted", "base", "l2w"):
        if cand in tokens:
            mode = cand
            break

    return policy, dataset, model, mode


def _infer_ft_from_rel_dir(rel_dir: str) -> str:
    parts = [p.lower() for p in rel_dir.split(os.sep) if p]
    if any("lora" in p for p in parts):
        return "lora"
    return "full"


def _read_peak_acc(eval_json_path: str) -> Optional[float]:
    """Return max accuracy from an *_eval.json if readable."""
    try:
        with open(eval_json_path, "r") as f:
            data = json.load(f)
        test = data.get("test")
        if isinstance(test, list) and test:
            return float(max(test))
    except Exception:
        pass
    return None


def discover_result_roots() -> List[str]:
    """List candidate results-root directories under project root."""
    # roots = set(["front-exps", "EXPS", "EXPS2"])
    
    try:
        for name in os.listdir(PROJECT_ROOT):
            full = os.path.join(PROJECT_ROOT, name)
            if not os.path.isdir(full):
                continue
            low = name.lower()
            if low.startswith("exps") or low == "front-exps" or low.startswith("exps_") or name.startswith("EXPS"):
                roots.add(name)
    except Exception:
        pass
    # Only keep existing ones
    roots = set(["front-exps"])
    return sorted([r for r in roots if os.path.isdir(os.path.join(PROJECT_ROOT, r))])


def scan_experiments(roots: List[str]) -> List[Dict]:
    """Walk each results-root and gather experiment records.

    Each record combines a .pth file (trained checkpoint) and its paired
    *_eval.json (if any) under the same directory, grouped by filename stem.
    """
    results: List[Dict] = []
    seen = set()

    for root_name in roots:
        root_abs = os.path.join(PROJECT_ROOT, root_name)
        if not os.path.isdir(root_abs):
            continue

        for dirpath, _dirs, files in os.walk(root_abs):
            pth_files = [f for f in files if f.endswith(".pth") and "_G_" not in f]
            eval_files = [f for f in files if f.endswith("_eval.json")]
            if not pth_files and not eval_files:
                continue

            rel_dir = os.path.relpath(dirpath, root_abs)
            ft = _infer_ft_from_rel_dir(rel_dir)

            # Only .pth files define experiment rows; eval files are
            # children of a .pth stem (prefix-matched), never standalone rows.
            stems = [f[:-4] for f in pth_files]

            for stem in sorted(stems):
                key = (root_name, rel_dir, stem)
                if key in seen:
                    continue
                seen.add(key)

                policy, dataset, model, mode = _parse_stem(stem)
                pth_name = stem + ".pth"
                has_model = True  # stem came from a .pth file

                # Find all eval files whose name starts with this stem
                matching_evals = [
                    f for f in eval_files
                    if f == stem + "_eval.json" or f.startswith(stem + "_")
                ]
                has_eval = len(matching_evals) > 0

                # Peak acc = best across all associated evals
                peak_acc = None
                for ef in matching_evals:
                    acc = _read_peak_acc(os.path.join(dirpath, ef))
                    if acc is not None:
                        peak_acc = acc if peak_acc is None else max(peak_acc, acc)

                mtime_candidates = []
                for f in [pth_name] + matching_evals:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        try:
                            mtime_candidates.append(os.path.getmtime(fp))
                        except Exception:
                            pass
                ts = max(mtime_candidates) if mtime_candidates else 0.0

                # Extract slim/exits suffix info for display (optional)
                slim = ""
                exits = ""
                if "slim" in stem:
                    slim = stem.split("slim", 1)[1].strip("_")
                if "exits" in stem:
                    exits = stem.split("exits", 1)[1].strip("_")

                results.append({
                    "root": root_name,
                    "rel_dir": rel_dir,
                    "stem": stem,
                    "policy": policy or "unknown",
                    "dataset": dataset or "unknown",
                    "model": model or "unknown",
                    "mode": mode or "-",
                    "ft": ft,
                    "slim": slim,
                    "exits": exits,
                    "has_model": bool(has_model),
                    "has_eval": bool(has_eval),
                    "peak_acc": float(peak_acc) if peak_acc is not None else float("nan"),
                    "last_modified_ts": ts,
                    "last_modified": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "",
                    "eval_json_path": os.path.join(dirpath, matching_evals[0]) if has_eval else "",
                })

    return results


# ---------------------------------------------------------------------------
# Dataset scanning (for Non-IID distribution viz)
# ---------------------------------------------------------------------------

def scan_datasets() -> List[Dict]:
    """Scan dataset/ for sub-directories that contain a config.json.

    Returns per-dataset dicts with a numpy matrix of shape (num_clients, num_classes)
    summarising each client's per-class sample counts, derived from the
    `Size of samples for labels in clients` field.
    """
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    results: List[Dict] = []
    if not os.path.isdir(dataset_dir):
        return results

    for name in sorted(os.listdir(dataset_dir)):
        p = os.path.join(dataset_dir, name)
        if not os.path.isdir(p):
            continue
        cfg_path = os.path.join(p, "config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            continue

        sizes = cfg.get("Size of samples for labels in clients", [])
        num_clients = int(cfg.get("num_clients", len(sizes)) or 0)
        num_classes = int(cfg.get("num_classes", 0) or 0)

        # Be robust to class indices outside declared num_classes
        observed_max_class = 0
        for client_info in sizes:
            for item in client_info:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        observed_max_class = max(observed_max_class, int(item[0]) + 1)
                    except Exception:
                        pass
        if observed_max_class > num_classes:
            num_classes = observed_max_class
        if num_clients <= 0:
            num_clients = len(sizes)

        matrix = np.zeros((max(num_clients, 1), max(num_classes, 1)), dtype=int)
        for ci, client_info in enumerate(sizes[:num_clients]):
            for item in client_info:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        cls, cnt = int(item[0]), int(item[1])
                    except Exception:
                        continue
                    if 0 <= cls < num_classes:
                        matrix[ci, cls] = cnt

        results.append({
            "name": name,
            "path": p,
            "num_clients": num_clients,
            "num_classes": num_classes,
            "non_iid": bool(cfg.get("non_iid", False)),
            "balance": bool(cfg.get("balance", False)),
            "partition": str(cfg.get("partition", "")),
            "matrix": matrix,
            "total_samples": int(matrix.sum()),
        })
    return results
