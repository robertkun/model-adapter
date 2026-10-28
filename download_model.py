from modelscope import snapshot_download

# 下载 Qwen3.5-2B 模型
# cache_dir 指定本地保存目录
model_dir = snapshot_download('Qwen/Qwen3.5-2B', cache_dir="./model")
print(f"✅ 模型下载到: {model_dir}")
