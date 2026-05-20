import streamlit as st
from modules import (
    home,
    generate,
    evaluate,
    settings,
    dataset_generate,
    model_download,
    structure,
    experiments,
    distribution,
)

st.set_page_config(page_title="FL-EE GUI", layout="wide")

# -----------------------------------------------------------------------------
# Design System — single CSS block covering sidebar + global styles
# -----------------------------------------------------------------------------
GLOBAL_CSS = '''
<style>
/* ================================================================
   PAGE BACKGROUND  +  remove top whitespace (hide Streamlit header)
================================================================ */
.stApp {
    background-color: #f4f6f9 !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
/* Hide Streamlit's top-bar decoration but keep the sidebar toggle */
[data-testid="stHeader"] > * {
    visibility: hidden;
}
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}
div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

/* ================================================================
   TYPOGRAPHY SCALE  (consistent across all pages)
================================================================ */
h1 {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #1e1e2e !important;
    letter-spacing: -0.3px;
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #1e1e2e !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.4rem !important;
}
h3 {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #1f2937 !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.25rem !important;
}
div[data-testid="stMarkdownContainer"] p {
    font-size: 15px;
    color: #374151;
    line-height: 1.7;
    margin: 0.3rem 0;
}
div[data-testid="stCaptionContainer"] p,
.stCaption, small {
    font-size: 13px !important;
    color: #6b7280 !important;
    line-height: 1.55 !important;
}

/* ================================================================
   METRIC CARDS
================================================================ */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.09);
}
div[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #ff8c00 !important;
}
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ================================================================
   BUTTONS
================================================================ */
.stButton > button {
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    height: 44px;
    transition: all 0.18s ease !important;
}
.stButton > button p {
    font-size: 15px !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* Secondary (default) — white with dark text and crisp border */
.stButton > button:not([kind="primary"]),
.stButton > button:not([kind="primary"]):hover,
.stButton > button:not([kind="primary"]):focus,
.stButton > button:not([kind="primary"]):active,
.stButton > button:not([kind="primary"]):focus-visible {
    background-color: #ffffff !important;
    background-image: none !important;
    color: #1e1e2e !important;
    border: 1.5px solid #d1d5db !important;
    outline: none !important;
}
.stButton > button:not([kind="primary"]) p,
.stButton > button:not([kind="primary"]) span,
.stButton > button:not([kind="primary"]) div {
    color: #1e1e2e !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #ff8c00 !important;
    background-color: rgba(255,140,0,0.08) !important;
}
.stButton > button:not([kind="primary"]):hover p,
.stButton > button:not([kind="primary"]):hover span,
.stButton > button:not([kind="primary"]):hover div {
    color: #ff8c00 !important;
}

/* Primary — flat orange (no gradient) with WHITE text on every state */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active,
.stButton > button[kind="primary"]:focus-visible {
    background-color: #ff8c00 !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1.5px solid #ff8c00 !important;
    outline: none !important;
    box-shadow: 0 2px 6px rgba(255,140,0,0.25) !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #e07b00 !important;
    background-image: none !important;
    border-color: #e07b00 !important;
    color: #ffffff !important;
    box-shadow: 0 6px 18px rgba(255,140,0,0.4) !important;
}
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span,
.stButton > button[kind="primary"]:hover div {
    color: #ffffff !important;
}

/* Download buttons inherit secondary style by default */
.stDownloadButton > button {
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    height: 44px !important;
}

/* ================================================================
   EXPANDERS
================================================================ */
div[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    overflow: hidden;
    margin-bottom: 0.75rem !important;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stExpander"] > details > summary {
    background: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #1e1e2e !important;
    padding: 12px 16px !important;
}
div[data-testid="stExpander"] > details[open] > summary {
    border-bottom: 1px solid #e5e7eb;
}

/* ================================================================
   TABS
================================================================ */
div[data-testid="stTabs"] button[data-testid="stTab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
}
div[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"] {
    color: #ff8c00 !important;
    border-bottom-color: #ff8c00 !important;
    font-weight: 600 !important;
}

/* ================================================================
   ALERTS
================================================================ */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 14px !important;
}

/* ================================================================
   DATAFRAME
================================================================ */
div[data-testid="stDataFrameContainer"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    overflow: hidden;
    background: #ffffff;
}

/* ================================================================
   DIVIDER
================================================================ */
hr {
    border-color: #e5e7eb !important;
    margin: 0.75rem 0 !important;
}

/* ================================================================
   FORM INPUTS
================================================================ */
.stTextInput input, .stNumberInput input {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-size: 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #ff8c00 !important;
    box-shadow: 0 0 0 2px rgba(255,140,0,0.15) !important;
}

/* ================================================================
   SIDEBAR — dark theme with orange accent
================================================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: none !important;
    box-shadow: 2px 0 16px rgba(0,0,0,0.2);
    max-width: 240px !important;
}
/* Zero out the horizontal padding on every sidebar container
   so the active nav band can go flush to both edges */
section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] .main,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="element-container"],
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stRadio > div,
section[data-testid="stSidebar"] .stRadio > label {
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
/* Title + caption stay padded (indent the brand area) */
section[data-testid="stSidebar"] [data-testid="stHeading"],
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    padding-left: 20px !important;
    padding-right: 20px !important;
}

/* Sidebar title "FL-EE" */
section[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: 2px;
    margin-bottom: 0 !important;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* Hide radio circles */
section[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}

/* Radio group — reset default gap */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px;
    width: 100%;
}

/* Nav item base — edge-to-edge fill when active */
section[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    color: #b0b0c8 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}
section[data-testid="stSidebar"] .stRadio label p {
    color: #b0b0c8 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    margin: 0 !important;
}
/* Hover */
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stRadio label:hover p {
    color: #ffffff !important;
}
/* Active / selected — full-width orange band with left accent bar */
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(255,140,0,0.28) 0%, rgba(255,140,0,0.10) 100%) !important;
    border-left-color: #ff8c00 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ffb347 !important;
    font-weight: 700 !important;
}

/* Section dividers — before items 4 (项目结构), 7 (训练与监控), 9 (设置) */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:nth-child(4),
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:nth-child(7),
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:nth-child(9) {
    margin-top: 10px !important;
    padding-top: 16px !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
}

/* Sidebar caption override */
section[data-testid="stSidebar"] .stCaption p {
    color: #5a5a7a !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    font-weight: 700;
}
</style>
'''
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# Page registry ---------------------------------------------------------------
PAGES = {
    "首页": home,
    "实验总览": experiments,
    "数据分布": distribution,
    "项目结构": structure,
    "模型下载": model_download,
    "数据集生成": dataset_generate,
    "训练与监控": generate,
    "模型评估": evaluate,
    "设置": settings,
}

PAGE_ICONS = {
    "首页":    "🏠",
    "实验总览": "📊",
    "数据分布": "📈",
    "项目结构": "🗂️",
    "模型下载": "📦",
    "数据集生成": "🗄️",
    "训练与监控": "🚀",
    "模型评估": "🔬",
    "设置":    "⚙️",
}


def main():
    """主入口函数"""
    # Cross-page navigation: other modules set st.session_state["_target_page"]
    if "_target_page" in st.session_state:
        target = st.session_state.pop("_target_page")
        if target in PAGES:
            st.session_state["_nav_radio"] = target

    with st.sidebar:
        st.title("FL-EE")
        st.caption("Federated Learning · Early Exit")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        selection = st.radio(
            "导航",
            list(PAGES.keys()),
            format_func=lambda k: f"{PAGE_ICONS.get(k, '')}  {k}",
            label_visibility="collapsed",
            key="_nav_radio",
        )

    PAGES[selection].show()


if __name__ == "__main__":
    main()
