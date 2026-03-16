
import torch
import os

def inspect_checkpoint(ckpt_dir='/root/autodl-tmp/val_ckpt/'):
    if not os.path.exists(ckpt_dir):
        print(f"错误: 目录 {ckpt_dir} 不存在")
        return

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpts:
        print("错误: 未找到任何 .ckpt 文件")
        return

    # 取最新的一个
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
    latest_ckpt = os.path.join(ckpt_dir, ckpts[0])
    
    print(f"\n{'='*20} 检查点深度探测 {'='*20}")
    print(f"文件: {latest_ckpt}")
    
    try:
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        keys = list(checkpoint.keys())
        
        print(f"\n[核心组件存在性报告]")
        print(f"- epoch: {'✅' if 'epoch' in keys else '❌'}")
        print(f"- global_step: {'✅' if 'global_step' in keys else '❌'}")
        print(f"- optimizer_states: {'✅' if 'optimizer_states' in keys else '❌'}")
        print(f"- lr_schedulers: {'✅' if 'lr_schedulers' in keys else '❌'}")
        
        if 'lr_schedulers' in keys:
            print(f"\n[Scheduler 状态]")
            sched_info = checkpoint['lr_schedulers']
            if isinstance(sched_info, list) and len(sched_info) > 0:
                # 尝试读取 step 数
                last_step = sched_info[0].get('_last_lr', '未知')
                print(f"- 已保存的最近 LR: {last_step}")
        else:
            print(f"\n⚠️ 警告: 检查点中缺失 lr_schedulers! 续训时 LR 将从初始值开始，导致‘猛冲’。")

    except Exception as e:
        print(f"读取失败: {e}")
    
    print(f"\n{'='*55}\n")

if __name__ == "__main__":
    inspect_checkpoint()
