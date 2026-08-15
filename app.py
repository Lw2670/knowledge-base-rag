# -*- coding: utf-8 -*-
import os
import streamlit as st
import rag
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="个人知识库问答", layout="wide")

st.title("个人知识库问答系统")
st.caption("LangChain + Chroma + 智谱 GLM · RAG 四步链路：加载 → 分块 → 向量化 → 检索生成")


@st.cache_resource(show_spinner=False)
def load_vectordb():
    """加载向量库；不存在则构建"""
    if os.path.exists(rag.CHROMA_DIR) and os.listdir(rag.CHROMA_DIR):
        return Chroma(persist_directory=rag.CHROMA_DIR, embedding_function=rag.get_embeddings())
    return rag.rebuild()


# 侧边栏：索引管理
with st.sidebar:
    st.header("索引管理")
    st.caption("笔记有更新后：点「刷新索引」，或重启网页自动同步")
    if st.button("刷新索引"):
        load_vectordb.clear()
        with st.spinner("重建索引中..."):
            rag.rebuild()
        st.success("索引已刷新，已同步最新笔记")

# 启动自动检测：已有索引但笔记更新了 → 自动重建
if os.path.exists(rag.CHROMA_DIR) and os.listdir(rag.CHROMA_DIR) and rag.needs_rebuild():
    load_vectordb.clear()
    with st.spinner("检测到笔记更新，自动重建索引..."):
        rag.rebuild()
    st.info("已自动同步最新笔记")


question = st.text_input("问你的知识库", placeholder="例如：我的赛事文案方法论是什么？")

if question.strip():
    vectordb = load_vectordb()

    # —— 智能体思考过程可视化 ——
    with st.status("智能体思考中...", expanded=True) as status:
        st.write("🔍 **① 理解问题**")
        st.write(f"　你的提问：{question}")

        st.write("📚 **② 检索相关笔记**")
        retrieved = rag.retrieve(vectordb, question, k=4)
        st.write(f"　在知识库中检索到 {len(retrieved)} 条最相关内容：")
        for i, s in enumerate(retrieved, 1):
            src = os.path.basename(s.metadata.get("source", "未知"))
            st.write(f"　· [{i}] {src}")

        st.write("🤖 **③ 生成答案**")
        status.update(label=f"思考完成 · 检索到 {len(retrieved)} 条笔记", state="complete")

    # —— 答案流式输出 ——
    st.markdown("### 回答")
    st.write_stream(rag.answer_stream(question, retrieved))

    st.markdown("### 引用来源")
    for s in retrieved:
        src = s.metadata.get("source", "未知")
        st.markdown(f"- `{src}`")
