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


def ask(vectordb, question, k=3):
    """4. 检索生成"""
    retrieved = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in retrieved])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.3,
    )
    prompt = (
        "请根据以下资料回答问题。如果资料中没有答案，就说不知道，不要编造。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "回答："
    )
    answer = llm.invoke(prompt)
    return answer.content, retrieved


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
