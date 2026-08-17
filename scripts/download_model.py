# -*- coding: utf-8 -*-
"""
下载 jina 中文嵌入模型并安装到 fastembed 本地缓存（国内可复现）。

为什么需要这个脚本：
- 系统默认用 jina-embeddings-v2-base-zh（768维中文嵌入）做向量化
- 该模型不在代码仓库里（体积 1.4GB），且 huggingface 国内网络不稳定
- 本脚本走阿里 ModelScope（魔搭）下载，再把文件按 fastembed 缓存格式摆好，
  之后 fastembed 会离线加载，不依赖网络。

用法：
    python scripts/download_model.py

说明：
- 模型来源：ModelScope（阿里魔搭社区），仓库 jinaai/jina-embeddings-v2-base-zh
- 安装位置：系统临时目录下的 fastembed_cache（与 fastembed 默认缓存一致）
- 运行后：用 rag.rebuild() 重建索引即生效（768 维）
"""
import os
import shutil
import sys
import tempfile

MODEL_ID = "jinaai/jina-embeddings-v2-base-zh"


def get_cache_dir():
    """与 fastembed.common.utils.define_cache_dir 保持一致"""
    return os.path.join(tempfile.gettempdir(), "fastembed_cache")


def ensure_modelscope():
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("安装 modelscope ...")
        os.system(f'"{sys.executable}" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope -q')
        from modelscope import snapshot_download
    return snapshot_download


def build_fastembed_cache(src, cache_dir):
    """把魔搭下载的文件整理成 fastembed 认识的 HuggingFace 缓存结构"""
    rev = "1234567890abcdef1234567890abcdef12345678"  # 合法 40 位 SHA
    repo_dir = os.path.join(cache_dir, f"models--{MODEL_ID.replace('/', '--')}")
    snap = os.path.join(repo_dir, "snapshots", rev)
    os.makedirs(os.path.join(snap, "onnx"), exist_ok=True)
    os.makedirs(os.path.join(snap, "1_Pooling"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)

    for fname in os.listdir(src):
        fp = os.path.join(src, fname)
        if os.path.isfile(fp) and fname.endswith((".json", ".txt")):
            shutil.copy2(fp, os.path.join(snap, fname))

    pool_cfg = os.path.join(src, "1_Pooling", "config.json")
    if os.path.exists(pool_cfg):
        shutil.copy2(pool_cfg, os.path.join(snap, "1_Pooling", "config.json"))

    # fastembed 期望 onnx/model.onnx（魔搭下载的是 model_fp16.onnx，重命名）
    onnx_src = os.path.join(src, "onnx", "model_fp16.onnx")
    if not os.path.exists(onnx_src):
        print("❌ 未找到 onnx/model_fp16.onnx，模型下载可能不完整，请重试")
        sys.exit(1)
    shutil.copy2(onnx_src, os.path.join(snap, "onnx", "model.onnx"))

    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write(rev)
    print(f"✅ 模型已安装到 fastembed 缓存：{repo_dir}")


def main():
    download = ensure_modelscope()
    print(f"从 ModelScope 下载 {MODEL_ID}（约 1.4GB，视网络而定）...")
    src = download(MODEL_ID)
    build_fastembed_cache(src, get_cache_dir())
    print("✅ 完成。运行 rag.rebuild() 即可用 jina（768 维）重建索引。")


if __name__ == "__main__":
    main()
