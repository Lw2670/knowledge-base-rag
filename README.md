# 个人知识库 RAG 问答系统

用 LangChain + Chroma + 智谱 GLM 搭建的个人知识库问答系统，文档源是自己的 Obsidian 笔记。

## 技术栈
- 框架：LangChain
- 向量库：Chroma（本地）
- Embedding：BGE-small-zh（fastembed，ONNX 轻量）
- 大模型：智谱 GLM-4-flash（免费）
- 文档源：./kb 下的 .md 笔记

## RAG 四步链路
1. 文档加载：读取 .md 文件（跳过隐藏目录）
2. 分块：RecursiveCharacterTextSplitter
3. 向量化入库：BGE 中文 embedding → Chroma
4. 检索生成：similarity_search 取 top-k → 智谱 GLM 生成带出处答案

## 快速开始
```bash
# 1. 配置密钥（config.py 已 gitignore）
# 2. 首次运行构建向量索引（会下载 BGE 模型）
python rag.py
# 3. 之后直接交互式提问
```

## 待办
- [x] Streamlit 网页界面
- [x] 索引自动更新
- [ ] 检索来源高亮展示
- [ ] 推 GitHub
