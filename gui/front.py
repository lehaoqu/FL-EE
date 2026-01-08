import streamlit as st
from modules import home, generate, evaluate, settings, dataset_generate, model_download

st.set_page_config(page_title="FL-EE GUI", layout="wide")

# -----------------------------------------------------------------------------
# CSS 样式区域 (保持你要求的卡片式导航)
# -----------------------------------------------------------------------------
SIDEBAR_CSS = '''
<style>
/* 隐藏 Sidebar 中 Radio 组件原本的圆圈单选按钮 */
section[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none;
}

/* 针对 Radio 选项的标签进行卡片化样式设置 */
section[data-testid="stSidebar"] .stRadio label {
    background-color: transparent;
    padding: 14px 24px;
    border-radius: 0px;
    margin-bottom: 0px;
    margin-left: -24px;
    margin-right: -24px;
    padding-left: 24px;
    padding-right: 24px;
    border: none;
    transition: all 0.2s ease;
    cursor: pointer;
    width: calc(100% + 48px);
    box-sizing: border-box;
    display: flex;
    justify-content: flex-start;  
    color: #555;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* 悬停状态 (Hover) */
section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: rgba(0, 0, 0, 0.06);
    color: #333;
    margin-bottom: 0px;
}

/* 选中状态 (Active/Checked) - 占满宽度的背景 */
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(255, 140, 0, 0.12) 0%, rgba(255, 165, 0, 0.08) 100%);
    color: #ff8c00;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0px;
}

/* 修正选中状态下内部文字颜色 */
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ff8c00 !important;
}

section[data-testid="stSidebar"] h1 {
    margin-bottom: 24px;
    font-size: 28px;
    font-weight: 700;
}
</style>
'''
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)


# 页面字典
PAGES = {
    "Home": home,
    "Models": model_download,
    "Dataset": dataset_generate,
    "Training": generate,
    "Evaluation": evaluate,
    "Settings": settings,
}


def main():
    """主入口函数"""
    with st.sidebar:
        st.title("FL-EE")
        selection = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    # 显示选中的页面
    page_module = PAGES[selection]
    page_module.show()


if __name__ == "__main__":
    main()
