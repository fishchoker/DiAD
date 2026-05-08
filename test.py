import random

import torchmetrics

from share import *

import pytorch_lightning as pl
import torch
import os
import argparse
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.model import create_model, load_state_dict
from utils.eval_helper import dump, log_metrics, merge_together, performances
from torch.nn import functional as F
import logging
import timm
from scipy.ndimage import gaussian_filter
import cv2
from utils.util import cal_anomaly_map, log_local, create_logger, setup_seed
from visa_dataloader import VisaDataset

# 参考 ldm/sgn 目录即包的导入方式
import dino.vision_transformer as vits

def get_all_layer_attentions(model, x):
    """
    通过 forward hook 收集 ViT 所有层的注意力矩阵。
    """
    all_attentions = []
    hooks = []

    def attn_hook(module, input, output):
        # DINO Attention.forward 返回 (x, attn)
        if isinstance(output, tuple):
            attn = output[1]   # [1, num_heads, N+1, N+1]
            all_attentions.append(attn.detach())

    for block in model.blocks:
        h = block.attn.register_forward_hook(attn_hook)
        hooks.append(h)

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return all_attentions

def attention_rollout(all_attentions, discard_ratio=0.9, head_fusion='mean'):
    """
    计算 Attention Rollout 显著性图。
    """
    num_patches = all_attentions[0].shape[-1] - 1  # 去掉 CLS token
    result = torch.eye(num_patches + 1).to(all_attentions[0].device)

    for attn in all_attentions:
        attn = attn[0]  # [num_heads, N+1, N+1]

        # 多头融合
        if head_fusion == 'mean':
            attn_fused = attn.mean(0)   # [N+1, N+1]
        elif head_fusion == 'max':
            attn_fused = attn.max(0).values
        elif head_fusion == 'min':
            attn_fused = attn.min(0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # 丢弃低注意力值（降噪）
        flat = attn_fused.flatten()
        threshold_val = torch.quantile(flat, discard_ratio)
        attn_fused[attn_fused < threshold_val] = 0.0

        # 残差连接：A_tilde = 0.5 * A + 0.5 * I
        identity = torch.eye(attn_fused.shape[0]).to(attn_fused.device)
        attn_tilde = 0.5 * attn_fused + 0.5 * identity

        # 每行重新归一化
        row_sum = attn_tilde.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_tilde = attn_tilde / row_sum

        # 跨层连乘累积
        result = torch.matmul(attn_tilde, result)

    # 取 CLS token（第0行）对所有 patch token（第1列以后）的注意力
    saliency = result[0, 1:]  # [num_patches]
    return saliency.cpu().numpy()

parser = argparse.ArgumentParser(description="DiAD")
parser.add_argument("--resume_path", default='./models/diad.ckpt')


args = parser.parse_args()

# Configs
resume_path = args.resume_path

batch_size = 1
logger_freq = 300
learning_rate = 1e-5
only_mid_control = True
evl_dir = "npz_result"
logger = create_logger("global_logger", "log/")

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
data_path = './training/MVTec-AD/mvtec_anomaly_detection/'
dataset = MVTecDataset('test', data_path)
# test_dataset = VisaDataset('test', data_path)


dataloader = DataLoader(dataset, num_workers=3, batch_size=batch_size, shuffle=True)
pretrained_model = timm.create_model("resnet50", pretrained=True, features_only=True)
pretrained_model = pretrained_model.cuda()
pretrained_model.eval()

# 加载 DINO 模型用于显著性检测（使用本地 dino 源码 + 本地权重）
dino_path = './models/dino_vits8.pth'
dino_model = vits.vit_small(patch_size=8)
if os.path.exists(dino_path):
    state_dict = torch.load(dino_path, map_location='cpu')
    dino_model.load_state_dict(state_dict, strict=False)
    print(f"Successfully loaded DINO weights from {dino_path}")
else:
    raise FileNotFoundError(
        f"DINO weights not found at '{dino_path}'.\n"
        f"Please download from:\n"
        f"https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth\n"
        f"and place it at '{dino_path}'."
    )
dino_model = dino_model.cuda().eval()

model.eval()
os.makedirs(evl_dir, exist_ok=True)
with torch.no_grad():
    for input in dataloader:
        input_img = input['jpg']
        input_features = pretrained_model(input_img.cuda())
        model = model.cuda()
        output= model.log_images_test(input)
        images = output
        log_local(images, input["filename"][0])
        output_img = images['samples']
        output_features = pretrained_model(output_img.cuda())
        input_features = input_features[1:4]
        output_features = output_features[1:4]

        # Calculate the anomaly score
        anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a')

        # DINO 提取逻辑：使用 Attention Rollout 获得更精细的显著性图
        img_for_dino = F.interpolate(input_img, size=(224, 224), mode='bilinear')
        
        # 1. 收集所有层的注意力矩阵
        all_layer_attentions = get_all_layer_attentions(dino_model, img_for_dino.cuda())
        
        # 2. 计算 Attention Rollout (discard_ratio=0.9 用于降噪)
        if all_layer_attentions:
            attentions_rollout = attention_rollout(all_layer_attentions, discard_ratio=0.9, head_fusion='mean')
        else:
            # Fallback: 如果 Hook 没拿到数据，退回到原来的最后一层均值方案
            print("Warning: Hook failed to capture attentions, falling back to last layer mean.")
            last_attn = dino_model.get_last_selfattention(img_for_dino.cuda())
            attentions_rollout = last_attn[0, :, 0, 1:].mean(0).cpu().numpy()

        # reshape 到空间图
        patch_size = 8  # dino_vits8
        h = w = 224 // patch_size  # 28×28
        saliency_map = attentions_rollout.reshape(h, w)
        saliency_map = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0)

        # 上采样到原图尺寸
        saliency_map = F.interpolate(
            saliency_map,
            size=(input_img.shape[-2], input_img.shape[-1]),
            mode='bilinear', align_corners=False
        )
        saliency_map = saliency_map[0, 0].cpu().numpy()

        # 归一化到 [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / \
                       (saliency_map.max() - saliency_map.min() + 1e-8)

        # 生成前景权重（软mask，保留部分背景避免过度抹除）
        foreground_weight = 0.2 + 0.8 * saliency_map  # 背景保留20%，前景保留100%

        # 加权异常分数
        anomaly_map = anomaly_map * foreground_weight

        anomaly_map = gaussian_filter(anomaly_map, sigma=5)
        anomaly_map = torch.from_numpy(anomaly_map)
        anomaly_map_prediction = anomaly_map.unsqueeze(dim=0).unsqueeze(dim=1)
        input["mask"] = input["mask"]

        root = os.path.join('log_image/')
        name = input["filename"][0][-7:-4]
        filename_feature = "{}-features.jpg".format(name)
        path_feature = os.path.join(root, input["filename"][0][:-7], filename_feature)
        pred_feature = anomaly_map_prediction.squeeze().detach().cpu().numpy()
        pred_feature = (pred_feature * 255).astype("uint8")
        pred_feature = Image.fromarray(pred_feature, mode='L')
        pred_feature.save(path_feature)

        #Heatmap
        anomaly_map_new = np.round(255 * (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()))
        anomaly_map_new = anomaly_map_new.cpu().numpy().astype(np.uint8)
        heatmap = cv2.applyColorMap(anomaly_map_new, colormap=cv2.COLORMAP_JET)
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        pixel_mean = torch.tensor(pixel_mean).unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
        pixel_std = torch.tensor(pixel_std).unsqueeze(1).unsqueeze(1)
        image = (input_img.squeeze() * pixel_std + pixel_mean) * 255
        image = image.permute(1, 2, 0).to('cpu').numpy().astype('uint8')
        image_copy = image.copy()
        out_heat_map = cv2.addWeighted(heatmap, 0.5, image_copy, 0.5, 0, image_copy)
        heatmap_name = "{}-heatmap.png".format(name)
        cv2.imwrite(root + input["filename"][0][:-7] + heatmap_name, out_heat_map)

        input['pred'] = anomaly_map_prediction
        input["output"] = output_img.cpu()
        input["input"] = input_img.cpu()

        output2 = input
        dump(evl_dir, output2)

evl_metrics = {'auc': [ {'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'}, {'name': 'f1px'}, {'name': 'f1sp'}]}
print("Gathering final results ...")
fileinfos, preds, masks = merge_together(evl_dir)
ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
log_metrics(ret_metrics, evl_metrics)