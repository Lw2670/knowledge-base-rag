# -*- coding: utf-8 -*-
"""
个人知识库 RAG 问答系统（最小版）
文档源：./kb 下的 .md 笔记
流程：文档加载 → 分块 → 向量化入库 → 检索生成
"""
import os
import time

# 国内网络加速：HuggingFace 镜像 + 禁用 Xet 存储（避免 401 错误）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

KB_DIR = r"./kb"     # 知识库目录
CHROMA_DIR = "./chroma_db"       # 向量库持久化目录
INDEX_STAMP = ".index_built_at"  # 索引构建时间戳文件（记录上次构建时间）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_docs():
    """1. 文档加载：遍历知识库所有 .md 文件，跳过隐藏目录"""
    md_files = []
    for root, dirs, files in os.walk(KB_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # 跳过 .workbuddy/.obsidian
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))
    docs = []
    for path in md_files:
        try:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"跳过 {os.path.basename(path)}: {e}")
    print(f"加载 {len(docs)} 个文档")
    return docs


def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


def build_index(docs):
    """2-3. 分块 + 向量化入库"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"切分 {len(chunks)} 个文本块")
    vectordb = Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_DIR)
    print("向量库构建完成")
    mark_built()
    return vectordb


def mark_built():
    """记录索引构建时间，用于判断笔记是否有更新"""
    with open(INDEX_STAMP, "w") as f:
        f.write(str(time.time()))


def latest_note_mtime():
    """返回知识库中最新的 .md 文件修改时间"""
    latest = 0
    for root, dirs, files in os.walk(KB_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".md"):
                latest = max(latest, os.path.getmtime(os.path.join(root, f)))
    return latest


def needs_rebuild():
    """判断笔记是否有更新：最新笔记修改时间 > 上次索引构建时间"""
    if not os.path.exists(INDEX_STAMP):
        return True
    try:
        built_at = float(open(INDEX_STAMP).read().strip())
    except Exception:
        return True
    return latest_note_mtime() > built_at


def rebuild():
    """清空旧索引并重建，返回新的向量库"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        for col in client.list_collections():
            client.delete_collection(col.name)
    except Exception:
        pass
    return build_index(load_docs())


def _make_llm(streaming=False):
    """创建大模型实例"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.3,
        streaming=streaming,
    )


def build_prompt(question, context, history_text=""):
    parts = ["请根据以下资料回答问题。如果资料中没有答案，就说不知道，不要编造。"]
    if history_text:
        parts.append(f"之前的对话：\n{history_text}")
    parts.append(f"资料：\n{context}")
    parts.append(f"问题：{question}")
    parts.append("回答：")
    return "\n\n".join(parts)


def _format_history(history):
    """把会话历史转成文本（最近 4 条）"""
    if not history:
        return ""
    recent = history[-4:]
    return "\n".join(
        [f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in recent]
    )


def _postprocess(scored, top_n=4, margin=0.2, max_dist=1.1):
    """
    检索结果后处理：按来源去重 → 绝对/相对阈值过滤 → 精选 top_n。
    scored: [(Document, distance)]，distance 越小越相关。
    margin: 相对阈值，与最优结果的允许差距比例。
    max_dist: 绝对阈值，最优匹配仍超过此距离 → 视为无相关内容。
    """
    if not scored:
        return []
    # 按来源去重：同一文件只保留最相关（distance 最小）的一条
    best_by_source = {}
    for doc, dist in scored:
        src = doc.metadata.get("source", "unknown")
        if src not in best_by_source or dist < best_by_source[src][1]:
            best_by_source[src] = (doc, dist)
    # 按距离排序
    ranked = sorted(best_by_source.values(), key=lambda x: x[1])
    # 绝对阈值：最优也过远 → 无相关内容
    if ranked[0][1] > max_dist:
        return []
    # 相对阈值过滤：与最优差距不超过 margin 才保留（滤掉明显不相关的）
    best = ranked[0][1]
    filtered = [(d, s) for d, s in ranked if s <= best * (1 + margin)]
    return [d for d, _ in filtered[:top_n]]


def retrieve(vectordb, question, k=8, top_n=4, margin=0.2, max_dist=1.1):
    """检索：取更多候选 → 去重 → 相关性过滤 → 精选 top_n"""
    scored = vectordb.similarity_search_with_score(question, k=k)
    return _postprocess(scored, top_n, margin, max_dist)


def retrieve_by_vector(vectordb, embedding, k=8, top_n=4, margin=0.2, max_dist=1.1):
    """按预计算向量检索（供搜索进度流程用，embedding 已算好）"""
    scored = vectordb.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
    return _postprocess(scored, top_n, margin, max_dist)


def build_query(question, history=None):
    """追问检测：问题很短且有历史时，拼接上一轮用户问题，帮助检索指代"""
    if history and len(question.strip()) < 12:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return prev[-1] + " " + question
    return question


def answer_stream(question, retrieved, history=None):
    """流式生成答案，逐段 yield 文本片段（供界面实时展示思考过程）"""
    context = "\n\n".join([d.page_content for d in retrieved])
    history_text = _format_history(history)
    for chunk in _make_llm(streaming=True).stream(build_prompt(question, context, history_text)):
        if chunk.content:
            yield chunk.content


def ask(vectordb, question, k=4, history=None):
    """完整流程（命令行用）：检索 + 生成，返回答案和来源"""
    query = build_query(question, history)
    retrieved = retrieve(vectordb, query, k)
    context = "\n\n".join([d.page_content for d in retrieved])
    history_text = _format_history(history)
    answer = _make_llm(streaming=False).invoke(build_prompt(question, context, history_text))
    return answer.content, retrieved


def chat_reply(text, history=None):
    """闲聊/直接回复：不检索知识库，直接让 LLM 自然回应"""
    history_text = _format_history(history)
    prompt = (
        "你是「个人知识库助手」。请自然、简洁地回应用户，语气友好。\n"
        + (f"之前的对话：\n{history_text}\n\n" if history_text else "")
        + f"用户：{text}\n"
        + "助手："
    )
    answer = _make_llm(streaming=False).invoke(prompt)
    return answer.content


def answer_stream_chat(text, history=None):
    """闲聊/直接回复（流式版），逐段 yield 文本"""
    history_text = _format_history(history)
    prompt = (
        "你是「个人知识库助手」。请自然、简洁地回应用户，语气友好。\n"
        + (f"之前的对话：\n{history_text}\n\n" if history_text else "")
        + f"用户：{text}\n"
        + "助手："
    )
    for chunk in _make_llm(streaming=True).stream(prompt):
        if chunk.content:
            yield chunk.content


if __name__ == "__main__":
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
        print("从已有向量库加载")
    else:
        vectordb = build_index(load_docs())

    print("\n===== 个人知识库问答（输入 exit 退出）=====")
    while True:
        q = input("\n你问：").strip()
        if q.lower() in ("exit", "quit", "退出"):
            break
        if not q:
            continue
        answer, sources = ask(vectordb, q)
        print(f"\n答：{answer}")
        print("\n--- 引用来源 ---")
        for s in sources:
            src = s.metadata.get("source", "未知")
            print(f"· {os.path.basename(src)}")
