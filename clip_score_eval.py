import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from tqdm import tqdm

# 设置模型路径
model_path = "/root/autodl-tmp/clip-vit-large-patch14"
data_root = "/root/autodl-tmp/mvtecad"

# MVTec-AD 类别列表
classes = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]

# 备选 Prompt 模板集合 (Ensemble Candidates)
def get_prompts(class_name):
    return [
        f"a photo of a normal {class_name} without defect",
        f"a normal {class_name} with intact surface, consistent texture, and clean edges",
        f"a photo of a normal {class_name} with smooth texture",
        f"a normal {class_name} with uniform color and consistent texture",
        f"a close-up photo of a normal {class_name} with clean edges",
        f"a flawless {class_name} in perfect condition",
        f"a standard {class_name} showing clear structural details"
    ]

def compute_clip_scores(model, processor, image_paths, prompts, device):
    if not image_paths:
        return None
        
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    # 提取图像特征
    inputs_img = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_img)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # 提取文本特征
    inputs_txt = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs_txt)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度矩阵 (Images x Prompts)
    similarities = (image_features @ text_features.T).cpu().numpy()
    
    # 每个 Prompt 的平均相似度
    mean_scores = similarities.mean(axis=0)
    return mean_scores

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model from {model_path}...")
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

    results = {}

    for cls in tqdm(classes, desc="Evaluating categories"):
        image_dir = os.path.join(data_root, cls, "train/good")
        if not os.path.exists(image_dir):
            print(f"Warning: {image_dir} not found, skipping...")
            continue
            
        # 每个类别随机取 5 张正常样本进行评估
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(image_dir, f) for f in image_files[:5]]
        
        prompts = get_prompts(cls)
        scores = compute_clip_scores(model, processor, image_paths, prompts, device)
        
        if scores is not None:
            # 排序获取得分最高的 Top-5
            prompt_scores = list(zip(prompts, scores))
            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            results[cls] = prompt_scores[:5]

    print("\n" + "="*50)
    print("TOP RECOMMENDED PROMPTS FOR EACH CLASS")
    print("="*50)
    
    for cls, top_prompts in results.items():
        print(f"\n[{cls.upper()}]")
        for i, (prompt, score) in enumerate(top_prompts):
            print(f"  {i+1}. Score: {score:.4f} | {prompt}")

if __name__ == "__main__":
    main()
