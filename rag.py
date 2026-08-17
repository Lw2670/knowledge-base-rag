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
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

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
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


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
        temperature=0.3,
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )


def build_prompt(question, context, history_text=""):
    """根据题型选择输出模式：项目/流程展示用结构化模板，定义/解释用自然叙述。"""
    q = question
    listing_kw = ["分别是什么", "有哪些项目", "有哪些作品", "我的项目", "我的作品",
                  "我的能力", "核心能力", "核心优势", "有什么作品", "有什么项目",
                  "都包括", "几个项目", "总结一下我的", "我会什么", "擅长什么",
                  "作品包括", "项目包括", "能力包括"]
    process_kw = ["怎么做", "步骤", "流程", "如何实现", "怎么落地", "怎么实现",
                  "怎么用", "怎么搭建", "如何搭建", "如何部署"]
    definition_kw = ["是什么", "方法论", "解释", "介绍", "理解", "原理", "为什么",
                     "怎么看", "意味着", "概念"]

    is_listing = any(k in q for k in listing_kw)
    is_process = any(k in q for k in process_kw)
    is_definition = any(k in q for k in definition_kw)

    # listing/process 优先于 definition（"我的项目分别是什么"是列举不是定义）
    use_template = is_listing or is_process

    if use_template:
        # 模式A：填空式结构化模板（模型对模板遵循好）
        parts = [
            "你是基于个人知识库的严谨问答助手。请**严格按以下模板**输出：",
            "",
            "### [项目/步骤名]",
            "- **🔧 技术亮点**：[用了什么技术 + 独特选型 + 与普通实现的差异，如独立从0到1 / 端到端全链路 / 工程化闭环 / 模型可插拔 / 隐私脱敏 / 三级数据兜底等，**说具体**]",
            "- **📊 量化成果**：[从资料里挖出可量化的信息（用户数/节省时间/规模/效果/时间线等）；资料确实没写数字，就用**具体事实**描述做了什么、效果如何，**不要写「无具体数字」这种空话**；严禁编造数字]",
            "- **💡 落地价值**：[解决什么痛点 / 谁受益 / 什么场景应用]",
            "",
            "**要求**：",
            "1. 整合提炼资料，先给整体结论再分项。",
            "2. 列举≥2项时结尾加「整体含金量」对比段（技术深度 / 业务价值 / 工程化 三维度）。",
            "3. 内容详实、有信息量，不要为了简短而遗漏要点。",
            "4. 禁止空话（「提升了能力」「证明了实力」替换成具体证据或删除）。",
        ]
    else:
        # 模式B：自然叙述（定义/解释型）
        parts = [
            "你是基于个人知识库的严谨问答助手。",
            "",
            "【输出模式：自然叙述】请用自然流畅、深入浅出的语言回答：",
            "",
            "要求：",
            "1. 整合提炼资料，先给核心要点再展开。",
            "2. **禁止硬套🔧/📊/💡三段式等固定框架**。用自然段落，必要时用小标题切分。",
            "3. 详略得当：定义/概念问题先给清晰定义+核心要素+实例/类比；概括/原因问题先结论后展开逻辑链。",
            "4. 严格基于资料，资料没有就说「未找到」。",
            "5. **数据以资料为准**：资料里有数字就用；没有就不硬编，用事实描述。",
            "6. 避免空洞表述（「提升了能力」「证明了实力」替换成具体证据或删除）。",
        ]

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
    """检索：取更多候选 → 去重 → 相关性过滤 → 精选 top_n"""
    scored = vectordb.similarity_search_with_score(question, k=k)
    return _postprocess(scored, top_n, margin, max_dist)


def retrieve_by_vector(vectordb, embedding, k=16, top_n=6, margin=0.35, max_dist=1.0):
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


def ask(vectordb, question, k=16, history=None):
    """完整流程（命令行用）：检索 + 生成，返回答案和来源"""
    query = build_query(question, history)
    retrieved = retrieve(vectordb, query, k=k)
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
