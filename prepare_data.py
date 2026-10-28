import json
import os

# 配置模型的身份信息
MY_NAME = "小胖"      # 模型的名字
MY_AUTHOR = "Robert"   # 作者/训练者

mlx_data = []

with open("./self-cognition/self_cognition.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        user_text = data["query"]
        assistant_text = data["response"]

        # 替换占位符
        if data["tag"] == "zh":
            assistant_text = assistant_text.replace("{{NAME}}", MY_NAME).replace("{{AUTHOR}}", MY_AUTHOR)
        else:
            assistant_text = assistant_text.replace("{{NAME}}", "XiaoPang").replace("{{AUTHOR}}", "Robert")

        mlx_data.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
        })

# 保存为 MLX 兼容格式
os.makedirs("./mlx_data", exist_ok=True)
with open("./mlx_data/train.jsonl", "w", encoding="utf-8") as f:
    for item in mlx_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 数据准备完成！共 {len(mlx_data)} 条数据")
