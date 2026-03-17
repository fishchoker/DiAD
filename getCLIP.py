# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
from transformers import CLIPTokenizer

# 设置镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

target_path = "/seu_nvme/home/yaoli/213221121/DiAD/DiAD/models/clip-vit-large-patch14"
os.makedirs(target_path, exist_ok=True)

print("正在下载 CLIP Tokenizer 相关文件...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokenizer.save_pretrained(target_path)
print(f"✅ 所有文件已保存至: {target_path}")
