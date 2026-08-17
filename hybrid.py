# -*- coding: utf-8 -*-
"""BM25 混合检索：jieba 分词 + BM25Okapi，与向量检索做 RRF 融合"""
import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

BM25_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.pkl")


def tokenize(text):
    return [w for w in jieba.lcut(text) if len(w.strip()) > 1]


def build_bm25(chunks, path=BM25_FILE):
    """构建 BM25 索引并保存（与向量库同步调用）"""
    corpus_tokens = [tokenize(c.page_content) for c in chunks]
    data = {
        "corpus_tokens": corpus_tokens,
        "texts": [c.page_content for c in chunks],
        "sources": [c.metadata.get("source", "") for c in chunks],
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return BM25Okapi(corpus_tokens)


def bm25_retrieve(query, k=16, path=BM25_FILE):
    """BM25 关键词检索，返回 [(Document, score)]，score>0 才保留"""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    bm25 = BM25Okapi(data["corpus_tokens"])
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    docs = []
    for i in top:
        if scores[i] > 0:
            docs.append(
                (
                    Document(
                        page_content=data["texts"][i],
                        metadata={"source": data["sources"][i]},
                    ),
                    scores[i],
                )
            )
    return docs


def _doc_key(doc):
    return (doc.metadata.get("source", ""), doc.page_content[:80])


def fuse_and_select(scored, bm25_docs, top_n=6, margin=0.35, max_dist=1.35, max_per_source=4):
    """
    RRF 融合向量+BM25 → 去重(每来源最多 max_per_source) → 阈值过滤 → top_n。
    scored: [(Document, distance)]，distance 越小越相关。
    """
    K = 60
    pool = {}  # key -> (rrf, doc, vec_dist)
    for rank, (doc, dist) in enumerate(scored):
        kk = _doc_key(doc)
        rrf, _, old_dist = pool.get(kk, (0.0, doc, None))
        pool[kk] = (rrf + 1.0 / (K + rank + 1), doc, old_dist if old_dist is not None else dist)
    for rank, (doc, _bs) in enumerate(bm25_docs):
        kk = _doc_key(doc)
        rrf, _, old_dist = pool.get(kk, (0.0, doc, None))
        pool[kk] = (rrf + 1.0 / (K + rank + 1), doc, old_dist)

    ranked = sorted(pool.values(), key=lambda x: -x[0])  # rrf 降序

    # 绝对阈值：取有向量距离的最优
    with_dist = [(d, s) for _, d, s in ranked if s is not None]
    if not with_dist:
        return []
    best = min(s for _, s in with_dist)
    if best > max_dist:
        return []

    # 按来源限流
    by_src = {}
    for _, doc, dist in ranked:
        src = doc.metadata.get("source", "unknown")
        by_src.setdefault(src, []).append((doc, dist))
    keep = []
    for src, items in by_src.items():
        items.sort(key=lambda x: x[1] if x[1] is not None else 999)
        keep.extend(items[:max_per_source])

    # 相对阈值过滤（有向量距离的）
    keep.sort(key=lambda x: x[1] if x[1] is not None else 999)
    filtered = [(d, s) for d, s in keep if s is None or s <= best * (1 + margin)]
    return [d for d, _ in filtered[:top_n]]
