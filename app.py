# -*- coding: utf-8 -*-
import os
import time
import streamlit as st
import rag
import intent
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


# 预热 embedding 模型（避免首次搜索时冷加载卡住）
@st.cache_resource
def warmup_embeddings():
    """启动时预热：触发一次空 embedding，让 BGE 模型从磁盘加载到内存"""
    rag.get_embeddings().embed_query("warmup")
    return "ready"

warmup_embeddings()  # 应用启动时执行一次


# ===== 意图处理分支（可配置）=====
def handle_command(text):
    """指令：触发系统操作"""
    if "刷新" in text or "重建" in text:
        load_vectordb.clear()
        rag.rebuild()
        return "索引已刷新，已同步最新笔记。"
    if "清空" in text:
        st.session_state.messages = []
        return "对话已清空，我们重新开始吧。"
    return "我没有理解这个指令。可以试试：「刷新索引」「清空对话」。"


def search_with_progress(vectordb, query, k=5):
    """
    搜索：带真实进度的分阶段检索。
    每个阶段对应真实的检索操作（理解→向量化→检索→整理）。
    注：不用 st.progress（它在 chat_message 内会冻结后续渲染），
       改用带 ✓ 箭头的 st.write 展示阶段，最稳。
    """
    st.write("① 理解查询...")
    time.sleep(0.15)

    st.write("② 向量化查询（生成查询向量）...")
    embedding = rag.get_embeddings().embed_query(query)  # 真实：embedding 计算
    time.sleep(0.15)

    st.write("③ 检索向量库（相似度匹配）...")
    results = vectordb.similarity_search_by_vector(embedding, k=k)  # 真实：向量检索
    time.sleep(0.15)

    st.write(f"④ 整理结果 ✓（找到 {len(results)} 条）")

    return results


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
        "我的四个 AI 项目分别是什么？",
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
    st.caption("笔记有更新后点刷新，或直接说「刷新索引」")
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
prompt = st.chat_input("输入你的问题、指令或闲聊...")
if not prompt and st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = st.session_state.messages[:-1]  # 不含刚加的用户消息

    # —— 意图识别（规则即时，模糊时 LLM + 候选意图）——
    result = intent.classify(prompt, llm=rag._make_llm(streaming=False))
    intent_label = intent.INTENT_META.get(result["intent"], result["intent"])

    with st.chat_message("assistant"):
        # 展示识别出的意图（即时反馈）
        conf_text = f"{result['confidence']:.0%}" if result["confidence"] >= 0.5 else "较低"
        st.caption(f"🎯 识别意图：{intent_label}（置信度 {conf_text}）")
        if result.get("candidates"):
            cand = " / ".join([intent.INTENT_META.get(c, c) for c in result["candidates"]])
            st.caption(f"🤔 也可能是：{cand}")

        # —— 按意图路由到处理分支 ——
        if result["intent"] == "command":
            answer = handle_command(prompt)
            st.markdown(answer)

        elif result["intent"] == "chitchat":
            answer = st.write_stream(rag.answer_stream_chat(prompt, history))

        elif result["intent"] == "search":
            try:
                retrieved = search_with_progress(load_vectordb(), prompt)
            except Exception as e:
                st.error(f"检索出错：{e}")
                retrieved = []
            if not retrieved:
                answer = "没有找到相关笔记。可以换个关键词试试。"
                st.markdown(answer)
            else:
                answer = f"找到 {len(retrieved)} 条相关笔记："
                st.markdown(answer)
                for i, s in enumerate(retrieved, 1):
                    fname = os.path.basename(s.metadata.get("source", "未知"))
                    with st.expander(f"[{i}] {fname}"):
                        st.markdown(s.page_content.strip())
                st.markdown("> 如果想让我基于这些笔记**总结或回答**，可以再说一句「帮我总结一下」")

        else:  # question（默认）：RAG 检索 + 生成
            vectordb = load_vectordb()
            query = rag.build_query(prompt, history)

            # 思考过程（用 st.write 常显，不用 st.status 避免完成后折叠隐藏步骤）
            st.write("🔍 **① 理解问题**")
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

            if not retrieved:
                answer = f"抱歉，我在知识库中没有找到与「{prompt}」相关的内容。\n\n你可以换个问法，或者先在知识库里补充相关笔记。"
                st.markdown(answer)
            else:
                try:
                    answer = st.write_stream(rag.answer_stream(prompt, retrieved, history))
                except Exception as e:
                    answer = f"生成答案时出错了：{e}"
                    st.error(answer)

            if retrieved:
                st.markdown("**引用来源**（点击展开查看原文）")
                for i, s in enumerate(retrieved, 1):
                    fname = os.path.basename(s.metadata.get("source", "未知"))
                    with st.expander(f"[{i}] {fname}"):
                        st.markdown(s.page_content.strip())

    # 保存 assistant 消息
    st.session_state.messages.append({"role": "assistant", "content": answer})
