# -*- coding: utf-8 -*-
"""用户意图识别模块

实时识别用户输入的意图（提问/搜索/指令/闲聊），特性：
- 规则快速分类：零延迟、零 LLM 调用
- 规则不确定时 LLM 分类：返回主意图 + 候选意图
- 低延迟：快路径命中即返回，慢路径仅在模糊输入时触发
"""
import re
import json

# 意图类别元信息（可扩展）
INTENT_META = {
    "question": "提问（检索知识库 + 生成回答）",
    "search": "搜索（只检索相关笔记）",
    "command": "指令（触发系统操作）",
    "chitchat": "闲聊（直接回复）",
}

# 规则快速路径（零延迟）：(意图, 置信度, [正则列表])
_RULES = [
    ("command", 0.9, [
        r"^(刷新|清空|重建|重置|重来)\s*(索引|对话|待办)?\s*[!！~～。]?$",
        r"(刷新|重建|清空)\s*(索引|对话|数据)",
    ]),
    ("chitchat", 0.9, [
        r"^(你好|您好|hi+|hello|在吗|谢谢|感谢|再见|拜拜)[!！~～。]?$",
        r"(你是谁|你叫什么|你能做什么|你会什么)",
    ]),
    ("search", 0.8, [
        r"^(找|搜索|查|列出|有哪些|有没有|帮我找|帮我查|帮我列出|看看|看一下|搜一下).*",
    ]),
    ("question", 0.7, [
        r"[?？]$",
        r"(什么|怎么|为什么|如何|哪些|多少|是不是|能否|可以吗|吗|呢|介绍一下|说说)",
    ]),
]


def classify_by_rules(text):
    """规则快速分类，返回 (intent, confidence)；不确定返回 (None, 0.0)"""
    t = text.strip()
    for intent, conf, patterns in _RULES:
        for p in patterns:
            if re.search(p, t, re.IGNORECASE):
                return intent, conf
    return None, 0.0


def _extract_json(text):
    """从 LLM 返回文本中提取 JSON 片段"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else "{}"


def classify_by_llm(text, llm):
    """LLM 分类（慢路径），返回 dict（含 intent/confidence/candidates）"""
    prompt = (
        "判断下面这句话的用户意图，从四类里选最合适的一个：\n"
        "- question：提问，需要检索知识库回答\n"
        "- search：搜索，想找相关笔记/资料\n"
        "- command：指令，想让系统执行操作\n"
        "- chitchat：闲聊，简单寒暄\n\n"
        "只输出 JSON，不要多余文字，格式：\n"
        '{"intent":"question","confidence":0.9,"candidates":["search"]}\n\n'
        f"用户输入：{text}"
    )
    try:
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_extract_json(content))
        return {
            "intent": data.get("intent", "question"),
            "confidence": float(data.get("confidence", 0.6)),
            "candidates": data.get("candidates", []),
            "method": "llm",
        }
    except Exception:
        return {"intent": "question", "confidence": 0.5, "candidates": [], "method": "llm-fallback"}


def classify(text, llm=None):
    """
    混合意图识别：先规则（零延迟），不确定再用 LLM（含候选意图）。
    返回 dict: {intent, confidence, candidates, method}
    """
    intent, conf = classify_by_rules(text)
    if intent is not None:
        return {"intent": intent, "confidence": conf, "candidates": [], "method": "rule"}
    if llm is not None:
        return classify_by_llm(text, llm)
    return {"intent": "question", "confidence": 0.5, "candidates": [], "method": "default"}
