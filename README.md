# 个人知识库问答系统（RAG）

> 把你的 Markdown 笔记，变成可对话的第二大脑。

一个基于 **RAG（检索增强生成）** 的知识库问答系统：读取你的 Markdown 笔记，用中文向量检索 + 大模型，回答你的问题——答案**带出处**，绝不凭空编造。

## 这是什么

传统的大模型会"凭空编造"答案。RAG 的思路是：先让你的笔记变成可检索的向量库，提问时**先检索出最相关的笔记，再让大模型基于这些笔记回答**。这样答案既准确、又有据可查。

四步链路：

```
文档加载 → 分块 → 向量化入库 → 检索生成
```

| 步骤 | 做什么 |
|---|---|
| 1. 文档加载 | 读取知识库里的 Markdown 笔记 |
| 2. 分块 | 把长文切成适合检索的小块 |
| 3. 向量化入库 | 用中文向量模型转成向量，存入本地向量库 Chroma |
| 4. 检索生成 | 提问 → 检索最相关笔记 → 大模型带出处回答 |

## 技术栈

| 组件 | 技术 |
|---|---|
| 框架 | LangChain |
| 向量库 | Chroma（本地，无需服务器） |
| 向量模型 | jina-embeddings-v2-base-zh（中文 768 维，本地运行） |
| 检索 | 向量 + BM25 混合（jieba 分词 + RRF 融合） |
| 大模型 | 智谱 GLM-4-flash（免费） |
| 界面 | Streamlit |

## 快速开始（4 步）

### 第 1 步：配置密钥

把 `config.py.example` 复制一份，改名为 `config.py`，填入你的智谱 API Key：

```python
LLM_API_KEY = "你的智谱 API Key"
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"
# 知识库目录（可选）：默认读取项目内 ./kb，指向你自己的 Obsidian 笔记仓库即可
# KB_DIR = r"D:\你的知识库"
```

> 智谱 API Key 获取：注册 [open.bigmodel.cn](https://open.bigmodel.cn)，免费领取 `glm-4-flash` 额度。

### 第 2 步：安装依赖

```bash
pip install -r requirements.txt
```

### 第 3 步：安装嵌入模型（重要）

系统用 **jina-embeddings-v2-base-zh**（768 维中文嵌入，约 1.4GB）做向量化。模型**不在代码仓库里**，需先执行一次下载脚本（国内走阿里 ModelScope 魔搭，不依赖 HuggingFace）：

```bash
python scripts/download_model.py
```

> - 脚本会把模型安装到 fastembed 本地缓存，之后全程离线运行，不再联网。
> - 模型来源：ModelScope 魔搭社区 `jinaai/jina-embeddings-v2-base-zh`（Apache-2.0）。
> - 首次运行系统时会自动重建索引（把 `kb/` 或你的知识库分块向量化），稍等片刻。

### 第 4 步：启动

```bash
# 网页版（推荐）
streamlit run app.py

# 或命令行版
python rag.py
```

启动后浏览器打开 `http://localhost:8501`，即可对知识库提问。

## 项目结构

```
├── rag.py              # RAG 核心（四步链路）
├── hybrid.py           # BM25 混合检索（jieba + rank_bm25）
├── intent.py           # 意图识别（规则快路径 + LLM 慢路径）
├── app.py              # Streamlit 网页界面
├── scripts/
│   └── download_model.py  # 嵌入模型下载脚本（ModelScope）
├── kb/                 # 示例知识库（3 篇 Markdown，可直接试用）
├── config.py           # 密钥配置（已 gitignore，不入库）
├── config.py.example   # 密钥配置模板
├── chroma_db/          # 本地向量库（自动生成，已 gitignore）
└── README.md
```

## 修改知识库内容后

系统会在启动时**自动检测**笔记是否有更新，有更新就重建索引；也可以在网页侧边栏点「刷新索引」手动更新。

## 常见问题

- **嵌入模型没下载/报错？** 先运行 `python scripts/download_model.py`，再启动。
- **想换大模型？** 改 `config.py` 里的 `LLM_MODEL` / `LLM_BASE_URL` 即可，系统对大模型解耦。
- **想换回更小的嵌入模型？** 改 `rag.py` 的 `get_embeddings()` 模型名（如 `BAAI/bge-small-zh-v1.5`），并把 `rag.py` 顶部 `HF_HUB_OFFLINE=1` 注释掉以便首次下载，重建索引即可。
