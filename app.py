# -*- coding: utf-8 -*-
import os
import streamlit as st
import rag
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="个人知识库问答", layout="wide")


@st.cache_resource(show_spinner=False)
def load_vectordb():
    """加载向量库；不存在则构建"""
    if os.path.exists(rag.CHROMA_DIR) and os.listdir(rag.CHROMA_DIR):
        return Chroma(persist_directory=rag.CHROMA_DIR, embedding_function=rag.get_embeddings())
    return rag.rebuild()


@st.cache_data
def count_notes():
    """统计知识库笔记数（轻量，不加载内容）"""
    n = 0
    for root, dirs, files in os.walk(rag.KB_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        n += sum(1 for f in files if f.endswith(".md"))
    return n


# 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []


# ===== 侧边栏：知识库概览 + 引导 + 管理 =====
with st.sidebar:
    st.title("个人知识库")
    st.caption("基于 RAG 的第二大脑问答")

    st.divider()
    st.subheader("知识库概览")
    c1, c2 = st.columns(2)
    c1.metric("笔记数", count_notes())
    c2.metric("对话轮数", len(st.session_state.messages) // 2)

    st.divider()
    st.subheader("试试这样问")
    examples = [
        "我的三个 AI 项目分别是什么？",
        "我的赛事文案方法论是什么？",
        "我的 RPA 自动化每年省多少时间？",
        "我转行 AI 的核心优势是什么？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

    st.divider()
    st.subheader("索引管理")
    st.caption("笔记有更新后点刷新")
    if st.button("刷新索引", use_container_width=True):
        load_vectordb.clear()
        with st.spinner("重建索引中..."):
            rag.rebuild()
        st.success("已同步最新笔记")

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ===== 主区：聊天 =====
st.title("个人知识库问答系统")
st.caption("问你的知识库，答案带出处，绝不凭空编造")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入（聊天框 或 侧边栏示例按钮）
prompt = st.chat_input("输入你的问题...")
if not prompt and st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        history = st.session_state.messages[:-1]  # 不含刚加的用户消息
        vectordb = load_vectordb()
        query = rag.build_query(prompt, history)

        # 智能体思考过程
        with st.status("智能体思考中...", expanded=True) as status:
            st.write("🔍 **① 理解问题**")
            st.write(f"　你的提问：{prompt}")
            st.write("📚 **② 检索相关笔记**")
            retrieved = rag.retrieve(vectordb, query, k=4)
            if not retrieved:
                st.write("　未在知识库中找到相关内容")
            else:
                st.write(f"　检索到 {len(retrieved)} 条最相关内容：")
                for i, s in enumerate(retrieved, 1):
                    src = os.path.basename(s.metadata.get("source", "未知"))
                    snippet = s.page_content.replace("\n", " ").strip()[:40]
                    st.write(f"　· [{i}] {src} — {snippet}...")
            st.write("🤖 **③ 生成答案**")
            status.update(label=f"思考完成 · 检索到 {len(retrieved)} 条笔记", state="complete")

        # 答案（流式 / 容错）
        if not retrieved:
            answer = f"抱歉，我在知识库中没有找到与「{prompt}」相关的内容。\n\n你可以换个问法，或者先在知识库里补充相关笔记。"
            st.markdown(answer)
        else:
            try:
                answer = st.write_stream(rag.answer_stream(prompt, retrieved, history))
            except Exception as e:
                answer = f"生成答案时出错了：{e}"
                st.error(answer)

        # 引用来源（可展开）
        if retrieved:
            st.markdown("**引用来源**（点击展开查看原文）")
            for i, s in enumerate(retrieved, 1):
                fname = os.path.basename(s.metadata.get("source", "未知"))
                with st.expander(f"[{i}] {fname}"):
                    st.markdown(s.page_content.strip())

    # 保存 assistant 消息
    st.session_state.messages.append({"role": "assistant", "content": answer})
