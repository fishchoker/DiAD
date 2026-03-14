
import sys
import os
import torch
import numpy as np

# 将当前目录加入路径以便导入
sys.path.append(os.getcwd())

from sgn.model import create_model
from mvtecad_dataloader import MVTecDataset

def verify_ensemble_logic():
    print("=== 开始验证 Prompt Ensemble 合法性 ===")
    
    # 1. 实例化模型 (不加载权重以节省时间)
    config_path = 'models/diad.yaml'
    print(f"正在从 {config_path} 创建模型...")
    model = create_model(config_path)
    
    # 2. 检查 CLASS_NAMES 和 class_prompts_list 对齐情况
    print("\n[检查 1: 物理清单对齐]")
    if not hasattr(model, 'CLASS_NAMES'):
        print("错误: 模型缺失 CLASS_NAMES 属性")
        return
    
    if not hasattr(model, 'class_prompts_list'):
        print("错误: 模型缺失 class_prompts_list 属性")
        return

    print(f"物理类别总数: {len(model.CLASS_NAMES)}")
    print(f"Prompt 列表总组数: {len(model.class_prompts_list)}")
    
    if len(model.CLASS_NAMES) != len(model.class_prompts_list):
        print(f"致命错误: 类别清单 ({len(model.CLASS_NAMES)}) 与 Prompt 组数 ({len(model.class_prompts_list)}) 不一致!")
    
    # 3. 检查输入结构 (嵌套列表问题诊断)
    print("\n[检查 2: 输入结构诊断]")
    sample_group = model.class_prompts_list[0]
    print(f"第 0 组 Prompt 结构: {type(sample_group)}")
    print(f"第 0 组 Prompt 内容: {sample_group}")
    
    # 模拟 get_learned_conditioning 调用
    print("\n尝试调用 get_learned_conditioning(sample_group)...")
    try:
        # 在 FrozenCLIPEmbedder.forward 中，tokenizer 期望的是 List[str]
        # 如果传入的是嵌套列表，这里会报错
        c_group = model.get_learned_conditioning(sample_group)
        print(f"成功获取 Embedding, 形状: {c_group.shape}")
    except Exception as e:
        print(f"调用失败! 错误信息: {e}")
        import traceback
        traceback.print_exc()

    # 4. 验证 Dataset 的 image_idx
    print("\n[检查 3: Dataset image_idx 验证]")
    try:
        dataset = MVTecDataset('train', '/root/autodl-tmp/mvtecad/', load_to_ram=False)
        print(f"Dataset CLASS_NAMES: {dataset.CLASS_NAMES}")
        
        # 检查是否一致
        if dataset.CLASS_NAMES != model.CLASS_NAMES:
            print("致命错误: Dataset 的 CLASS_NAMES 与 Model 的 CLASS_NAMES 顺序不一致!")
        else:
            print("通过: Dataset 与 Model 的物理清单完全一致。")
            
        # 采样验证映射
        sample_item = dataset.data[0]
        cls_name = sample_item['clsname']
        expected_idx = str(dataset.CLASS_NAMES.index(cls_name))
        actual_idx = dataset.label_to_idx[cls_name]
        print(f"类别 '{cls_name}' -> 预期索引: {expected_idx}, 实际映射: {actual_idx}")
        
        if expected_idx != actual_idx:
             print("错误: label_to_idx 映射与物理顺序不符!")
        else:
             print("通过: label_to_idx 映射与物理顺序完全对齐。")

    except Exception as e:
        print(f"验证 Dataset 时出错 (可能路径不对): {e}")

if __name__ == "__main__":
    verify_ensemble_logic()
