import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# ✅ 直接从项目已有模块导入，不重复定义
from sgn.prompts import CATEGORY_PROMPTS

# ==================== 配置区域 ====================
DATA_ROOT       = './training/MVTec-AD/mvtec_anomaly_detection/'
TEST_JSON       = './training/MVTec-AD/test.json'
LOCAL_CLIP_PATH = './models/clip-vit-large-patch14'
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES     = 20
# ==================================================

# 其余代码完全不变...
ANOMALY_TEMPLATES = [
    "a photo of a damaged {cls}.",
    "a photo of a defective {cls} with scratches or holes.",
    "a photo of a broken {cls}."
]


# ===================== 模型加载 =====================

def load_clip():
    print(f"正在从本地加载 CLIP 模型: {LOCAL_CLIP_PATH}...")
    model     = CLIPModel.from_pretrained(LOCAL_CLIP_PATH).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_PATH)
    print("CLIP 模型加载完成。")
    return model, processor


# ===================== 编码工具函数 =====================

@torch.no_grad()
def encode_texts(model, processor, texts: list) -> torch.Tensor:
    """返回 L2 归一化后的文本特征 [N, D]"""
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    feats = model.get_text_features(**inputs).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_image(model, processor, image_path: str) -> torch.Tensor:
    """返回 L2 归一化后的图像特征 [1, D]"""
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    feats  = model.get_image_features(**inputs).float()
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats


# ===================== 数据加载 =====================

def get_dataset_info():
    with open(TEST_JSON, 'rt') as f:
        return [json.loads(line) for line in f]


def build_cls_data(dataset_info):
    """按类别整理正常/异常图像路径"""
    cls_data = {}
    for item in dataset_info:
        cls = item['clsname']
        if cls not in cls_data:
            cls_data[cls] = {'normal': [], 'anomaly': []}
        path = os.path.join(DATA_ROOT, item['filename'])
        if item['label'] == 0:
            cls_data[cls]['normal'].append(path)
        else:
            cls_data[cls]['anomaly'].append(path)
    return cls_data


# ===================== 方法二：提示词多样性检验 =====================

def test_prompt_diversity(model, processor, category_prompts):
    print("\n" + "="*60)
    print("方法二：提示词多样性检验 (Cosine Similarity between prompts)")
    print("="*60)
    print(f"{'Category':12s} | {'Avg Similarity':14s} | {'Status':20s}")
    print("-" * 55)

    for cls, prompts in category_prompts.items():
        if not prompts or len(prompts) < 2:
            continue

        feats      = encode_texts(model, processor, prompts)   # [N, D]
        sim_matrix = feats @ feats.T                            # [N, N]
        mask       = ~torch.eye(len(prompts), dtype=torch.bool, device=DEVICE)
        avg_sim    = sim_matrix[mask].mean().item()

        if avg_sim > 0.98:
            status = "⛔ 几乎重复，建议重写"
        elif avg_sim > 0.95:
            status = "⚠️  过于相似"
        else:
            status = "✅ 多样性良好"

        print(f"{cls:12s} | {avg_sim:14.4f} | {status}")


# ===================== 方法一 & 三：基于图像的评估 =====================

def test_prompts_with_images(model, processor, dataset_info, category_prompts):
    print("\n" + "="*60)
    print("方法一 & 方法三：基于图像的零样本评估")
    print("  Normal Gap  = 正常图与正常提示词相似度 - 正常图与异常提示词相似度")
    print("  Anomaly Gap = 异常图与正常提示词相似度 - 异常图与异常提示词相似度")
    print("  期望：Normal Gap > 0 > Anomaly Gap，差距越大越好")
    print("  Zero-shot AUC：仅凭文本语义引导，不训练扩散模型时的检测上限")
    print("="*60)
    print(f"{'Category':12s} | {'Normal Gap':10s} | {'Anomaly Gap':11s} | {'AUC':8s} | {'Gap Diff':8s}")
    print("-" * 62)

    cls_data = build_cls_data(dataset_info)
    all_aucs = []

    for cls, prompts in category_prompts.items():
        if cls not in cls_data or not prompts:
            continue

        # 预计算文本特征
        normal_feats  = encode_texts(model, processor, prompts)           # [N, D]
        anomaly_texts = [t.format(cls=cls) for t in ANOMALY_TEMPLATES]
        anomaly_feats = encode_texts(model, processor, anomaly_texts)     # [M, D]

        # 集成正常特征（用于 AUC 打分）
        ensemble_feat = normal_feats.mean(0, keepdim=True)
        ensemble_feat = ensemble_feat / ensemble_feat.norm(dim=-1, keepdim=True)  # [1, D]

        def get_scores(img_path):
            """返回 (gap, auc_score)"""
            img_feat  = encode_image(model, processor, img_path)          # [1, D]
            s_normal  = (img_feat @ normal_feats.T).mean().item()
            s_anomaly = (img_feat @ anomaly_feats.T).mean().item()
            s_auc     = (img_feat @ ensemble_feat.T).item()
            return s_normal - s_anomaly, s_auc

        n_paths   = cls_data[cls]['normal'][:MAX_SAMPLES]
        a_paths   = cls_data[cls]['anomaly'][:MAX_SAMPLES]

        n_results = [get_scores(p) for p in tqdm(n_paths, desc=f"{cls:12s} normal",  leave=False)]
        a_results = [get_scores(p) for p in tqdm(a_paths, desc=f"{cls:12s} anomaly", leave=False)]

        avg_n_gap = np.mean([r[0] for r in n_results]) if n_results else 0.0
        avg_a_gap = np.mean([r[0] for r in a_results]) if a_results else 0.0
        gap_diff  = avg_n_gap - avg_a_gap  # 越大越好

        # AUC：正常图 s_auc 高 → 异常分数取负值
        y_true   = [0] * len(n_results) + [1] * len(a_results)
        y_scores = [-r[1] for r in n_results] + [-r[1] for r in a_results]

        try:
            auc = roc_auc_score(y_true, y_scores)
            if auc < 0.5:        # 自动修正方向
                auc = 1.0 - auc
        except ValueError:
            auc = 0.5

        all_aucs.append(auc)
        print(f"{cls:12s} | {avg_n_gap:10.4f} | {avg_a_gap:11.4f} | {auc:8.4f} | {gap_diff:8.4f}")

    print("-" * 62)
    print(f"{'Mean':12s} | {'-':10s} | {'-':11s} | {np.mean(all_aucs):8.4f} |")
    print("\n[评估指南]")
    print("  Gap Diff > 0.02 : 提示词对该类别有区分能力")
    print("  AUC > 0.70      : 零样本语义引导有效，值得保留")
    print("  AUC < 0.55      : 提示词对该类别几乎无效，建议重写或放弃")
    print("  Diversity > 0.98: 提示词过于重复，CLIP 集成无收益")


# ===================== 主入口 =====================

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"错误：找不到数据集目录 {DATA_ROOT}，请检查配置。")
        exit(1)

    if not os.path.exists(TEST_JSON):
        print(f"错误：找不到测试集 JSON {TEST_JSON}，请检查配置。")
        exit(1)

    model, processor = load_clip()
    dataset_info     = get_dataset_info()

    test_prompt_diversity(model, processor, CATEGORY_PROMPTS)
    test_prompts_with_images(model, processor, dataset_info, CATEGORY_PROMPTS)