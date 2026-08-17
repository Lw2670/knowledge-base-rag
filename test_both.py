# -*- coding: utf-8 -*-
"""自检：分别测试「搜索」和「提问」两条分支是否都正常渲染"""
from playwright.sync_api import sync_playwright

def test(page, query):
    chat = page.locator('[data-testid="stChatInput"] textarea')
    chat.fill(query)
    chat.press("Enter")
    page.wait_for_timeout(12000)
    body = page.inner_text("body")
    return body

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
    page.wait_for_selector('[data-testid="stChatInput"]', timeout=30000)
    print("✅ 页面加载")

    # 测试 1：搜索指令
    body1 = test(page, "帮我列出所有项目")
    print("\n=== 搜索指令「帮我列出所有项目」===")
    for kw in ["识别意图", "理解查询", "找到"]:
        print(("✅" if kw in body1 else "❌"), f"含「{kw}」")

    # 测试 2：普通提问
    body2 = test(page, "我的 RPA 自动化每年省多少时间")
    print("\n=== 普通提问「我的 RPA 自动化每年省多少时间」===")
    for kw in ["识别意图", "思考", "检索", "生成", "回答", "1500"]:
        print(("✅" if kw in body2 else "❌"), f"含「{kw}」")
    # 打印提问分支实际显示了什么
    print("--- 提问分支页面片段 ---")
    for line in body2.split("\n"):
        s = line.strip()
        if s and any(k in s for k in ["意图", "思考", "检索", "生成", "回答", "省", "小时", "错误"]):
            print("  >", s[:90])

    browser.close()
