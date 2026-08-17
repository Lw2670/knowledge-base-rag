# -*- coding: utf-8 -*-
import os
import time
import html
import streamlit as st
import rag
import intent
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="个人知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom theme / UI CSS
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Design Tokens ──────────────────────────────────────────────────────── */
:root {
    --brand: #0f6e56;
    --brand-hover: #0d5d48;
    --brand-soft: #e1f5ee;
    --brand-foreground: #ffffff;
    --bg: #f7f9f8;
    --fg: #0f1f1c;
    --surface: #ffffff;
    --surface-subtle: #f0f4f3;
    --border: #dfe7e4;
    --muted: #5f6f6b;
    --code-bg: #111827;
    --code-text: #a7f3d0;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --radius-pill: 999px;
    --shadow-1: 0 1px 2px rgba(15,31,28,0.04);
    --shadow-2: 0 8px 24px -8px rgba(15,31,28,0.12);
    --shadow-3: 0 24px 60px -20px rgba(15,31,28,0.18);
    --ease-out: cubic-bezier(0.25, 1, 0.5, 1);
    --ease-in-out: cubic-bezier(0.45, 0, 0.55, 1);
}

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Inter", "PingFang SC", "Microsoft YaHei", "Noto Sans SC", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background: var(--bg) !important;
    color: var(--fg) !important;
}

.stAppHeader, footer, #MainMenu { visibility: hidden; }

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
    color: var(--fg) !important;
    font-weight: 700 !important;
    font-size: 26px !important;
    letter-spacing: -0.025em !important;
    line-height: 1.2 !important;
}
h2 {
    color: var(--fg) !important;
    font-weight: 600 !important;
    font-size: 20px !important;
    letter-spacing: -0.015em !important;
}
h3 {
    color: var(--fg) !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
p, .stMarkdown {
    color: var(--fg) !important;
    line-height: 1.7 !important;
}
.stCaption {
    color: var(--muted) !important;
    font-size: 13px !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--fg) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 20px 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--fg) !important;
    font-weight: 500 !important;
    transition: border-color 0.2s var(--ease-out),
                background 0.2s var(--ease-out),
                color 0.2s var(--ease-out),
                transform 0.15s var(--ease-out) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--brand) !important;
    color: var(--brand) !important;
    background: var(--brand-soft) !important;
    transform: translateX(2px);
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: translateX(0);
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 14px !important;
    transition: border-color 0.2s var(--ease-out),
                box-shadow 0.2s var(--ease-out) !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--brand) !important;
    box-shadow: var(--shadow-1) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricValue"] {
    color: var(--brand) !important;
    font-weight: 700 !important;
    font-size: 28px !important;
    line-height: 1.1 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: transform 0.15s var(--ease-out),
                background 0.2s var(--ease-out),
                border-color 0.2s var(--ease-out),
                color 0.2s var(--ease-out),
                box-shadow 0.2s var(--ease-out) !important;
}
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--brand) !important;
    border-color: var(--brand) !important;
    color: var(--brand-foreground) !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--brand-hover) !important;
    border-color: var(--brand-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px -4px rgba(15,110,86,0.3) !important;
}
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
}
.stButton > button[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--fg) !important;
}
.stButton > button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--brand) !important;
    color: var(--brand) !important;
    background: var(--brand-soft) !important;
}

/* ── Chat Input ─────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-1) !important;
    transition: border-color 0.2s var(--ease-out),
                box-shadow 0.2s var(--ease-out) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px rgba(15,110,86,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--fg) !important;
    font-size: 15px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}

/* ── Chat Messages ──────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    margin-bottom: 20px !important;
    animation: msgSlideIn 0.35s var(--ease-out) both;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    color: var(--brand) !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    color: var(--muted) !important;
}

@keyframes msgSlideIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.chat-user, .chat-assistant {
    padding: 14px 20px;
    line-height: 1.7;
    max-width: 88%;
    width: fit-content;
    font-size: 15px;
    word-break: break-word;
}
.chat-user {
    background: var(--brand);
    color: var(--brand-foreground);
    border-radius: 18px 18px 4px 18px;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(15,110,86,0.2);
}
.chat-assistant {
    background: var(--surface);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    box-shadow: var(--shadow-1);
}
.chat-assistant p {
    color: var(--fg) !important;
    margin: 0 0 8px 0 !important;
}
.chat-assistant p:last-child {
    margin-bottom: 0 !important;
}
.chat-meta {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
    font-weight: 500;
    padding: 4px 10px;
    display: inline-block;
    background: var(--surface-subtle);
    border-radius: var(--radius-pill);
    border: 1px solid var(--border);
}

/* ── Source Cards ───────────────────────────────────────────────────────── */
.source-card {
    background: var(--surface-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin: 10px 0;
    transition: border-color 0.2s var(--ease-out),
                box-shadow 0.2s var(--ease-out),
                transform 0.15s var(--ease-out);
}
.source-card:hover {
    border-color: var(--brand);
    box-shadow: var(--shadow-1);
    transform: translateY(-1px);
}
.source-title {
    color: var(--brand);
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.source-body {
    color: var(--fg);
    font-size: 14px;
    line-height: 1.7;
}

/* ── Compact Source List (one line per source) ───────────────────────── */
.source-list {
    margin: 6px 0 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.source-line {
    font-size: 12px;
    color: var(--muted);
    background: var(--surface-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 4px 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Welcome Card ───────────────────────────────────────────────────────── */
.welcome-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 56px 40px;
    text-align: center;
    max-width: 680px;
    margin: 48px auto 0;
    box-shadow: var(--shadow-2);
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--brand), #34d399, var(--brand));
    opacity: 0.6;
}
.welcome-card h2 {
    font-size: 28px !important;
    margin-bottom: 12px !important;
    color: var(--fg) !important;
    font-weight: 700 !important;
}
.welcome-card p {
    color: var(--muted) !important;
    font-size: 15px !important;
    margin-bottom: 0 !important;
    max-width: 440px;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* ── Code Blocks ────────────────────────────────────────────────────────── */
pre {
    background: var(--code-bg) !important;
    border-radius: var(--radius-sm) !important;
    padding: 16px !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
code {
    font-family: "SF Mono", "JetBrains Mono", Consolas, "Courier New", monospace !important;
    color: var(--code-text) !important;
    font-size: 0.9em !important;
}

/* ── Status / Steps ─────────────────────────────────────────────────────── */
[data-testid="stStatus"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-1) !important;
    transition: border-color 0.2s var(--ease-out) !important;
}
[data-testid="stStatus"]:hover {
    border-color: var(--brand) !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: background 0.2s var(--ease-out) !important;
}
.streamlit-expanderHeader:hover {
    background: var(--surface-subtle) !important;
}
.streamlit-expanderContent {
    border: none !important;
    padding: 16px !important;
    background: var(--surface) !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── Info / Success / Warning Blocks ────────────────────────────────────── */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    font-size: 14px !important;
}
.stSuccess {
    background: var(--brand-soft) !important;
    border-color: var(--brand) !important;
}
.stInfo {
    background: var(--surface-subtle) !important;
    border-color: var(--border) !important;
}

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
    transition: background 0.2s var(--ease-out);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--muted);
}

/* ── Reduced Motion ─────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_vectordb():
    """加载向量库：按 .index_built_at 时间戳自动检测重建（外部 rebuild 也能立即生效）"""
    stamp = rag.INDEX_STAMP
    current_mtime = os.path.getmtime(stamp) if os.path.exists(stamp) else 0
    cached = st.session_state.get("_vdb")
    if cached and cached[0] == current_mtime:
        return cached[1]
    if os.path.exists(rag.CHROMA_DIR) and os.listdir(rag.CHROMA_DIR):
        vdb = Chroma(persist_directory=rag.CHROMA_DIR, embedding_function=rag.get_embeddings())
    else:
        vdb = rag.rebuild()
    st.session_state["_vdb"] = (current_mtime, vdb)
    return vdb


@st.cache_data
def count_notes():
    """统计知识库笔记数（轻量，不加载内容）"""
    n = 0
    for root, dirs, files in os.walk(rag.KB_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        n += sum(1 for f in files if f.endswith(".md"))
    return n


@st.cache_resource
def warmup_embeddings():
    """启动时预热：触发一次空 embedding，让 BGE 模型从磁盘加载到内存"""
    rag.get_embeddings().embed_query("warmup")
    return "ready"


warmup_embeddings()


def handle_command(text):
    """指令：触发系统操作"""
    if "刷新" in text or "重建" in text:
        rag.rebuild()
        st.session_state.pop("_vdb", None)  # 清除向量库缓存，强制下次重新加载
        return "索引已刷新，已同步最新笔记。"
    if "清空" in text:
        st.session_state.messages = []
        return "对话已清空，我们重新开始吧。"
    return "我没有理解这个指令。可以试试：「刷新索引」「清空对话」。"


def search_with_progress(vectordb, query, k=8, top_n=5):
    """带阶段反馈的向量检索（检索更多 → 去重 → 相关性过滤 → 精选）"""
    embedding = rag.get_embeddings().embed_query(query)
    results = rag.retrieve_by_vector(vectordb, embedding, query, k=k, top_n=top_n)
    return results


def render_user_message(text):
    st.markdown(f'<div class="chat-user">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_assistant_container():
    """返回一个可写入 assistant 气泡的占位，调用方继续在其内部写入内容"""
    return st.markdown('<div class="chat-assistant">', unsafe_allow_html=True)

def split_answer_sections(text):
    """把模型输出按【思考过程】【最终答案】【来源引用】切分为三段"""
    import re
    parts = re.split(r"(【思考过程】|【最终答案】|【来源引用】|【资料来源引用】)", text)
    sections = {"思考过程": "", "最终答案": "", "来源引用": ""}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip("【】")
        val = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if "思考" in key:
            sections["思考过程"] = (sections["思考过程"] + "\n" + val).strip()
        elif "答案" in key or "最终" in key:
            sections["最终答案"] = (sections["最终答案"] + "\n" + val).strip()
        elif "来源" in key or "引用" in key:
            sections["来源引用"] = (sections["来源引用"] + "\n" + val).strip()
    return sections


def render_structured_answer(text):
    """渲染三段式回答：🧠思考过程(可折叠) → 最终答案 → 📚来源引用"""
    sec = split_answer_sections(text)
    if not any(sec.values()):
        # 模型没按格式输出，原样展示
        st.markdown(f'<div class="chat-assistant">{html.escape(text)}</div>', unsafe_allow_html=True)
        return
    if sec["思考过程"]:
        with st.expander("🧠 思考过程（模型推理步骤）", expanded=True):
            st.markdown(sec["思考过程"])
    if sec["最终答案"]:
        st.markdown(sec["最终答案"])
    if sec["来源引用"]:
        st.markdown(f'<div class="source-list">📚 {html.escape(sec["来源引用"])}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "我的四个 AI 项目分别是什么？",
    "我的赛事文案方法论是什么？",
    "我转行 AI 的核心优势是什么？",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 个人知识库")
    st.caption("基于 RAG 的第二大脑问答")
    st.divider()

    st.markdown("**知识库概览**")
    c1, c2 = st.columns(2)
    c1.metric("笔记数", count_notes())
    c2.metric("对话轮数", len(st.session_state.messages) // 2)

    st.divider()
    st.markdown("**试试这样问**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending = ex
            st.rerun()

    st.divider()
    st.markdown("**索引管理**")
    st.caption("笔记更新后可点击刷新")
    if st.button("🔄 刷新索引", use_container_width=True):
        with st.spinner("重建索引中..."):
            rag.rebuild()
        st.session_state.pop("_vdb", None)  # 清除缓存，强制下次重新加载
        st.success("已同步最新笔记")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("<h1 style='margin-bottom:4px;'>个人知识库问答系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#5f6f6b; margin-top:0;'>问你的知识库，答案带出处，绝不凭空编造</p>", unsafe_allow_html=True)

# Empty-state welcome
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>把笔记变成可对话的第二大脑</h2>
            <p>在下方输入问题，或点击一个示例开始体验。系统会先检索你的 Markdown 笔记，再让大模型基于笔记内容生成回答。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.markdown("<div style='text-align:center; margin-top:16px;'>", unsafe_allow_html=True)
        for ex in EXAMPLES:
            if st.button(ex, key=f"main_ex_{ex}"):
                st.session_state.pending = ex
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            # Assistant content is already HTML-escaped by generator; wrap safely
            st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
prompt = st.chat_input("输入你的问题、指令或闲聊...")
if not prompt and st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    # User turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_user_message(prompt)

    history = st.session_state.messages[:-1]

    # Intent classification
    result = intent.classify(prompt, llm=rag._make_llm(streaming=False))
    intent_label = intent.INTENT_META.get(result["intent"], result["intent"])

    with st.chat_message("assistant"):
        answer = ""

        # Intent badge
        conf_text = f"{result['confidence']:.0%}" if result["confidence"] >= 0.5 else "较低"
        st.markdown(
            f"<div class='chat-meta'>🎯 识别意图：{html.escape(intent_label)}（置信度 {conf_text}）</div>",
            unsafe_allow_html=True,
        )
        if result.get("candidates"):
            cand = " / ".join([intent.INTENT_META.get(c, c) for c in result["candidates"]])
            st.markdown(f"<div class='chat-meta'>🤔 也可能是：{html.escape(cand)}</div>", unsafe_allow_html=True)

        # ---- command ----
        if result["intent"] == "command":
            answer = handle_command(prompt)
            st.markdown(f'<div class="chat-assistant">{html.escape(answer)}</div>', unsafe_allow_html=True)

        # ---- chitchat ----
        elif result["intent"] == "chitchat":
            with st.status("正在闲聊...", expanded=False):
                pass
            answer = st.write_stream(rag.answer_stream_chat(prompt, history))
            st.markdown(f'<div class="chat-assistant">{html.escape(answer)}</div>', unsafe_allow_html=True)

        # ---- search ----
        elif result["intent"] == "search":
            with st.status("正在检索笔记...", expanded=True) as status:
                try:
                    retrieved = search_with_progress(load_vectordb(), prompt)
                except Exception as e:
                    st.error(f"检索出错：{e}")
                    retrieved = []
                status.update(label=f"检索完成 · 找到 {len(retrieved)} 条笔记", state="complete")

            if not retrieved:
                answer = "没有找到相关笔记。可以换个关键词试试。"
                st.markdown(f'<div class="chat-assistant">{html.escape(answer)}</div>', unsafe_allow_html=True)
            else:
                answer = f"找到 {len(retrieved)} 条相关笔记："
                st.markdown(f'<div class="chat-assistant">{html.escape(answer)}</div>', unsafe_allow_html=True)
                for i, s in enumerate(retrieved, 1):
                    fname = os.path.basename(s.metadata.get("source", "未知"))
                    with st.expander(f"📄 [{i}] {fname}"):
                        st.markdown(s.page_content.strip())
                st.info("如果想让我基于这些笔记**总结或回答**，可以再说一句「帮我总结一下」")

        # ---- question (default) ----
        else:
            vectordb = load_vectordb()
            query = rag.build_query(prompt, history)

            with st.status("正在基于知识库生成答案...", expanded=True) as status:
                st.write("🔍 理解问题")
                st.write("📚 检索相关笔记")
                retrieved = rag.retrieve(vectordb, query)
                if not retrieved:
                    st.write("　未在知识库中找到相关内容")
                else:
                    st.write(f"　检索到 {len(retrieved)} 条最相关内容：")
                    for i, s in enumerate(retrieved, 1):
                        src = os.path.basename(s.metadata.get("source", "未知"))
                        snippet = s.page_content.replace("\n", " ").strip()[:40]
                        st.write(f"　· [{i}] {src} — {snippet}...")
                status.update(label="检索完成，开始生成答案", state="running")

            if not retrieved:
                answer = f"抱歉，我在知识库中没有找到与「{prompt}」相关的内容。\n\n你可以换个问法，或者先在知识库里补充相关笔记。"
                st.markdown(f'<div class="chat-assistant">{html.escape(answer)}</div>', unsafe_allow_html=True)
            else:
                try:
                    # 流式预览（实时滚动），结束后切成三段式可视化
                    ph = st.empty()
                    chunks = []
                    for chunk in rag.answer_stream(prompt, retrieved, history):
                        chunks.append(chunk)
                        ph.markdown("".join(chunks))
                    answer = "".join(chunks)
                    ph.empty()
                except Exception as e:
                    answer = f"生成答案时出错了：{e}"
                    st.error(answer)

                render_structured_answer(answer)

        # Persist assistant turn
        st.session_state.messages.append({"role": "assistant", "content": answer})
