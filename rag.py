# -*- coding: utf-8 -*-
"""
个人知识库 RAG 问答系统
文档源：KB_DIR 指向的 Markdown 知识库目录（默认 ./kb，可在 config.py 中覆盖）
流程：文档加载 → 分块 → 向量化入库 → 检索生成
"""
import os
import re
import time
from functools import lru_cache

# 国内网络加速：HuggingFace 镜像 + 禁用 Xet 存储（避免 401 错误）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型本地缓存加载，避免联网超时
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
import hybrid

try:
    from config import KB_DIR   # 知识库目录（可在 config.py 中自定义，指向你的笔记）
except ImportError:
    KB_DIR = "./kb"             # 默认：项目内示例知识库目录

CHROMA_DIR = "./chroma_db"       # 向量库持久化目录
INDEX_STAMP = ".index_built_at"  # 索引构建时间戳文件（记录上次构建时间）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


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
    return FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-zh")


def _split_docs_markdown(docs):
    """按 Markdown 标题结构分块：先按标题切段（保留标题作上下文），段内过长再按字符切块"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        # 只在 # / ## 标题前切段（h3 跟随父级），段内过长再按字符切块
        sections = re.split(r"(?m)^(?=#{1,2} )", doc.page_content)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            sec_doc = Document(page_content=sec, metadata={"source": src})
            if len(sec) <= CHUNK_SIZE * 1.3:
                new_docs.append(sec_doc)
            else:
                new_docs.extend(text_splitter.split_documents([sec_doc]))
    return new_docs


def build_index(docs):
    """2-3. 分块（Markdown 结构感知）+ 向量化入库"""
    chunks = _split_docs_markdown(docs)
    print(f"切分 {len(chunks)} 个文本块")
    hybrid.build_bm25(chunks)  # 同步构建 BM25 关键词索引
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


@lru_cache(maxsize=4)
def _make_llm(streaming=False):
    """创建大模型实例（按 streaming 参数缓存，避免重复构造；带超时与自动重试）"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,  # 知识库问答用 0，减少幻觉
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )


_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """# 角色与规则
你是基于参考资料作答的问答助手，必须严格遵循下面流程，**先输出完整思考过程，再输出最终答案**。
【约束铁则】
1. 所有信息只能来自【参考上下文】，上下文没有的内容，不要编造，直接说明“参考资料中无相关信息”；
2. 禁止引入你的内置知识库信息，优先采信参考原文；
3. 如果参考资料存在冲突，要在思考环节指出冲突点；
4. 思考过程和最终答案必须明确分区，格式不能乱。
{history_text}

# 输出固定格式（严格遵守）
【思考过程】
一步步分析：
1. 先拆解用户核心问题是什么，明确需要从参考资料里找哪些信息；
2. 核对检索到的参考上下文，定位相关原文片段；
3. 判断资料是否足够回答问题、是否存在信息缺失/矛盾；
4. 规划如何组织答案、哪些内容不能说。

【最终答案】
清晰、精炼、引用原文依据作答；资料不足时如实说明。

# 参考上下文
{context}

# 用户问题
{question}"""
)

MAX_CONTEXT_CHARS = 12000  # 召回上下文上限，防止超模型上下文窗口


def build_prompt(question, context, history_text=""):
    """生成提示词文本（兼容旧调用，实际链路走 LCEL 模板）"""
    return _PROMPT_TEMPLATE.format(
        question=question, context=context, history_text=history_text
    )


def _format_context(retrieved, max_chars=MAX_CONTEXT_CHARS):
    """拼接检索上下文，超长截断防爆窗"""
    parts, total = [], 0
    for d in retrieved:
        c = d.page_content
        if total + len(c) > max_chars:
            parts.append(c[: max_chars - total])
            break
        parts.append(c)
        total += len(c)
    return "\n\n".join(parts)


def _build_chain(streaming=False):
    """LCEL RAG 链：输入 {question, context, history_text} → 三段式回答"""
    return (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "history_text": lambda x: x.get("history_text", ""),
        }
        | _PROMPT_TEMPLATE
        | _make_llm(streaming=streaming)
    )


def _format_history(history):
    """把会话历史转成文本（最近 4 条），加隔离标记防模型模仿历史中的旧格式"""
    if not history:
        return ""
    recent = history[-4:]
    msgs = [f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}" for m in recent]
    header = "【以下是之前的对话，仅作为上下文参考，**不要模仿其中的排版格式】"
    return chr(10).join([header] + msgs)

def _postprocess(scored, top_n=6, margin=0.35, max_dist=1.0, max_per_source=4):
    """
    检索结果后处理：按来源限流 → 绝对/相对阈值过滤 → 精选 top_n。
    scored: [(Document, distance)]，distance 越小越相关。
    margin: 相对阈值，与最优结果的允许差距比例。
    max_dist: 绝对阈值，最优匹配仍超过此距离 → 视为无相关内容。
    max_per_source: 每个来源文件最多保留几条（防单篇刷屏，同时允许有料文件多贡献）。
    """
    if not scored:
        return []
    # 按来源限流：同一文件最多保留 max_per_source 条最相关 chunk
    by_source = {}
    for doc, dist in scored:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append((doc, dist))
    keep = []
    for src, items in by_source.items():
        items.sort(key=lambda x: x[1])
        keep.extend(items[:max_per_source])
    # 按距离排序
    ranked = sorted(keep, key=lambda x: x[1])
    # 绝对阈值：最优也过远 → 无相关内容
    if ranked[0][1] > max_dist:
        return []
    # 相对阈值过滤：与最优差距不超过 margin 才保留（滤掉明显不相关的）
    best = ranked[0][1]
    filtered = [(d, s) for d, s in ranked if s <= best * (1 + margin)]
    return [d for d, _ in filtered[:top_n]]


def retrieve(vectordb, question, k=16, top_n=6, margin=0.35, max_dist=1.0):
    """混合检索：向量 + BM25 关键词 → RRF 融合 → 去重过滤 → 精选 top_n"""
    scored = vectordb.similarity_search_with_score(question, k=k)
    bm25_docs = hybrid.bm25_retrieve(question, k=k)
    return hybrid.fuse_and_select(scored, bm25_docs, top_n, margin, max_dist)


def retrieve_by_vector(vectordb, embedding, query_text, k=16, top_n=6, margin=0.35, max_dist=1.0):
    """按预计算向量检索（供搜索进度流程用，embedding 已算好）+ BM25 融合"""
    scored = vectordb.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
    bm25_docs = hybrid.bm25_retrieve(query_text, k=k)
    return hybrid.fuse_and_select(scored, bm25_docs, top_n, margin, max_dist)


def build_query(question, history=None):
    """追问检测：问题很短且有历史时，拼接上一轮用户问题，帮助检索指代"""
    if history and len(question.strip()) < 12:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return prev[-1] + " " + question
    return question


def answer_stream(question, retrieved, history=None):
    """流式生成答案（LCEL 链），逐段 yield 文本片段"""
    context = _format_context(retrieved)
    history_text = _format_history(history)
    chain = _build_chain(streaming=True)
    for chunk in chain.stream(
        {"question": question, "context": context, "history_text": history_text}
    ):
        if chunk.content:
            yield chunk.content


def ask(vectordb, question, k=16, history=None):
    """完整流程（命令行用）：检索 + 生成，返回答案和来源"""
    query = build_query(question, history)
    retrieved = retrieve(vectordb, query, k=k)
    context = _format_context(retrieved)
    history_text = _format_history(history)
    chain = _build_chain(streaming=False)
    answer = chain.invoke(
        {"question": question, "context": context, "history_text": history_text}
    )
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