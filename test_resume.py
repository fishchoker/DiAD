
import torch
import os
import sys
from pytorch_lightning import Trainer
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

def test_resume_state(ckpt_path, config_path='models/diad.yaml'):
    print(f"\n{'='*20} 检查点恢复测试 {'='*20}")
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到检查点文件 {ckpt_path}")
        return

    # 1. 加载检查点原始数据
    print(f"正在读取检查点: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 2. 提取 Epoch 和 Global Step
    epoch = checkpoint.get('epoch', '未知')
    global_step = checkpoint.get('global_step', '未知')
    
    print(f"\n[状态确认]")
    print(f"- 检查点记录的 Epoch: {epoch}")
    print(f"- 检查点记录的 Step: {global_step}")
    
    if 'optimizer_states' in checkpoint:
        print(f"- 优化器状态: 已找到 (续训将恢复学习率和动量)")
    else:
        print(f"- 警告: 未找到优化器状态，续训可能无法完全对齐状态")

    # 3. 验证评价模型过滤逻辑
    state_dict = checkpoint.get('state_dict', {})
    unexpected_keys = [k for k in state_dict.keys() if k.startswith('pretrained_model')]
    if unexpected_keys:
        print(f"- 冗余键值: 发现 {len(unexpected_keys)} 个评价模型权重 (将在加载时自动过滤)")
    else:
        print(f"- 冗余键值: 无 (检查点已精简)")

    # 4. 模拟加载逻辑
    print(f"\n[逻辑验证]")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    # 模拟 PL 的加载过程
    try:
        # 这里会触发我们之前写的 on_load_checkpoint 拦截器
        # 注意：PL 的 on_load_checkpoint 是在 Trainer 运行 fit 时调用的
        # 我们这里直接手动调用它来验证过滤逻辑
        from ldm.models.diffusion.ddpm import LatentDiffusion
        if hasattr(model, 'on_load_checkpoint'):
            model.on_load_checkpoint(checkpoint)
            
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"结果: 模型权重成功装载，未发生崩溃。")
    except Exception as e:
        print(f"结果: 加载失败! 错误信息: {e}")

    print(f"\n{'='*55}\n")
    print("提示: 若要正式续训并【跳过初始验证】，请使用以下命令启动:")
    print(f"python train.py --resume_from {ckpt_path} --num_sanity_val_steps 0")
    print("\n注: --num_sanity_val_steps 0 是 PyTorch Lightning 跳过启动前验证的关键参数。")

if __name__ == "__main__":
    # 默认检查数据盘路径，如果不存在则检查当前目录
    target_ckpt = '/root/autodl-tmp/val_ckpt/last.ckpt'
    if not os.path.exists(target_ckpt):
        # 尝试寻找目录下的任意 ckpt
        ckpt_dir = '/root/autodl-tmp/val_ckpt/'
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpts:
                target_ckpt = os.path.join(ckpt_dir, sorted(ckpts, reverse=True)[0])
    
    test_resume_state(target_ckpt)
