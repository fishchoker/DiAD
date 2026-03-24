import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sgn.prompts import CATEGORY_PROMPTS
from tqdm import tqdm

# =================配置区域=================
DATA_ROOT = './training/MVTec-AD/mvtec_anomaly_detection/'
TEST_JSON = './training/MVTec-AD/test.json'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-L/14" # 使用与项目一致的大模型
# ==========================================

def load_clip():
    print(f"正在加载 CLIP 模型 {CLIP_MODEL}...")
    model, preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    return model, preprocess

def get_dataset_info():
    with open(TEST_JSON, 'rt') as f:
        data = [json.loads(line) for line in f]
    return data

# 方法二：提示词间语义距离检验
def test_prompt_diversity(model, category_prompts):
    print("\n--- 方法二：提示词多样性检验 (Diversity) ---")
    results = {}
    for cls, prompts in category_prompts.items():
        if not prompts or all(p == "" for p in prompts):
            continue
        
        tokens = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        
        # 计算两两相似度
        sim_matrix = feats @ feats.T
        # 提取非对角线元素
        mask = ~torch.eye(len(prompts), dtype=torch.bool, device=DEVICE)
        off_diag = sim_matrix[mask]
        avg_sim = off_diag.mean().item()
        results[cls] = avg_sim
        
        status = "⚠️ 过于相似" if avg_sim > 0.95 else "✅ 多样性良好"
        print(f"{cls:12s}: 平均相似度={avg_sim:.4f} {status}")
    return results

# 方法一 & 方法三：基于实际图像的质量评估
def test_prompts_with_images(model, preprocess, dataset_info, category_prompts):
    print("\n--- 方法一 & 方法三：基于图像的零样本评估 ---")
    
    # 构造通用的异常提示词
    anomaly_prompts_template = [
        "a photo of a damaged {cls}.",
        "a photo of a defective {cls} with scratches or holes.",
        "a photo of a broken {cls}."
    ]
    
    # 按类别组织数据
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

    print(f"{'Category':12s} | {'Normal Gap':10s} | {'Anomaly Gap':10s} | {'Zero-shot AUC':10s}")
    print("-" * 55)

    all_auc_scores = []

    for cls, prompts in category_prompts.items():
        if cls not in cls_data or not prompts:
            continue
        
        # 准备文本特征
        normal_tokens = clip.tokenize(prompts).to(DEVICE)
        anomaly_texts = [t.format(cls=cls) for t in anomaly_prompts_template]
        anomaly_tokens = clip.tokenize(anomaly_texts).to(DEVICE)
        
        with torch.no_grad():
            normal_txt_feat = model.encode_text(normal_tokens)
            normal_txt_feat /= normal_txt_feat.norm(dim=-1, keepdim=True)
            
            anomaly_txt_feat = model.encode_text(anomaly_tokens)
            anomaly_txt_feat /= anomaly_txt_feat.norm(dim=-1, keepdim=True)
            
            # 集成特征 (用于 AUC 计算)
            ensemble_normal_feat = normal_txt_feat.mean(0, keepdim=True)
            ensemble_normal_feat /= ensemble_normal_feat.norm(dim=-1, keepdim=True)

        def get_gap(img_path):
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = model.encode_image(img)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                
                # 计算与正常/异常文本的相似度
                s_normal = (img_feat @ normal_txt_feat.T).mean().item()
                s_anomaly = (img_feat @ anomaly_txt_feat.T).mean().item()
                
                # AUC 使用的原始分数 (与正常提示词的相似度)
                s_auc = (img_feat @ ensemble_normal_feat.T).item()
            return s_normal - s_anomaly, s_auc

        # 计算 Gap
        # 为了速度，每个类别最多随机取 20 张正常和 20 张异常
        n_paths = cls_data[cls]['normal'][:20]
        a_paths = cls_data[cls]['anomaly'][:20]
        
        n_results = [get_gap(p) for p in n_paths]
        a_results = [get_gap(p) for p in a_paths]
        
        n_gaps = [r[0] for r in n_results]
        a_gaps = [r[0] for r in a_results]
        
        avg_n_gap = np.mean(n_gaps) if n_gaps else 0
        avg_a_gap = np.mean(a_gaps) if a_gaps else 0
        
        # 方法三：AUC 评估
        # 准备 AUC 计算的标签和分数
        y_true = [0] * len(n_results) + [1] * len(a_results)
        # 分数：相似度越高越可能是正常，所以取负值作为异常得分
        y_scores = [-r[1] for r in n_results] + [-r[1] for r in a_results]
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            all_auc_scores.append(auc)
        except:
            auc = 0.5

        print(f"{cls:12s} | {avg_n_gap:10.4f} | {avg_a_gap:10.4f} | {auc:10.4f}")

    if all_auc_scores:
        print("-" * 55)
        print(f"{'Mean':12s} | {'-':10s} | {'-':10s} | {np.mean(all_auc_scores):10.4f}")

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"错误：找不到数据集目录 {DATA_ROOT}，请检查配置。")
    else:
        model, preprocess = load_clip()
        dataset_info = get_dataset_info()
        
        # 执行方法二
        test_prompt_diversity(model, CATEGORY_PROMPTS)
        
        # 执行方法一 & 三
        test_prompts_with_images(model, preprocess, dataset_info, CATEGORY_PROMPTS)
        
        print("\n[评估指南]")
        print("1. Normal Gap 应显著大于 Anomaly Gap (Gap 越大说明提示词越能区分该类别的正常态)。")
        print("2. Diversity 相似度应在 0.8~0.9 之间。如果 > 0.98，建议更换提示词增加差异化。")
        print("3. Zero-shot AUC 反映了该提示词在不训练扩散模型时，仅靠语义引导能达到的上限。")
