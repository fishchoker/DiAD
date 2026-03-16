import os
import tempfile
import argparse

# 强制指定临时文件目录到数据盘
os.makedirs('/root/autodl-tmp/tmp', exist_ok=True)
os.environ['TMPDIR'] = '/root/autodl-tmp/tmp'
tempfile.tempdir = '/root/autodl-tmp/tmp'

# 设置 CUDA 显存分配配置，减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 设置 HuggingFace 镜像源以加速国内下载 (针对 timm/clip 等)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from share import *
import torch
import cv2
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict
from visa_dataloader import VisaDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 开启 TensorFloat-32 (TF32) 以加速 5090 上的矩阵运算
torch.set_float32_matmul_precision('high')

# 禁用 OpenCV 多线程，防止与 DataLoader 的多进程产生冲突，导致 CPU 效率下降
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# 设置随机种子以保证实验的可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置 cuDNN 在确定性模式下运行，确保每次运行结果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置参数
resume_path = './models/diad.ckpt'  # 预训练模型路径
ckpt_dir = './val_ckpt/'            # 训练检查点保存目录

# 命令行参数解析
parser = argparse.ArgumentParser(description="DiAD Training Script")
parser.add_argument("--resume_from", type=str, default=None, help="Explicitly specify a checkpoint file to resume from.")
parser.add_argument("--version", type=int, default=None, help="Explicitly specify the lightning_logs version to append to.")
args = parser.parse_args()

setup_seed(1)
# 自动检测最新的检查点用于断点续训
def find_latest_checkpoint(dir_path):
    if not os.path.exists(dir_path):
        return None
    ckpts = [f for f in os.listdir(dir_path) if f.endswith('.ckpt')]
    if not ckpts:
        return None
    # 按修改时间排序，取最新的一个 (通常是最好的模型)
    ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
    return os.path.join(dir_path, ckpts[0])

# 自动检测最新的日志版本，确保断点续训时曲线连续
def find_latest_version(log_dir):
    if not os.path.exists(log_dir):
        return None
    versions = [f for f in os.listdir(log_dir) if f.startswith('version_')]
    if not versions:
        return None
    # 提取数字并取最大值
    version_nums = [int(v.split('_')[1]) for v in versions]
    return max(version_nums)

# 优先级：命令行参数 > 自动检测
last_ckpt = args.resume_from if args.resume_from else find_latest_checkpoint(ckpt_dir)
latest_version = args.version if args.version is not None else find_latest_version("lightning_logs/")

if last_ckpt:
    target_version_str = f"version_{latest_version}" if latest_version is not None else "new"
    print(f"\n" + "="*50)
    print(f"检测到历史检查点: {last_ckpt}")
    print(f"状态: 准备断点续训 (保留 Epoch/Step/Optimizer)")
    print(f"日志: 将追加至 {target_version_str}")
    print("="*50 + "\n")
else:
    target_version_str = f"version_{latest_version}" if args.version is not None else "new"
    print(f"\n" + "="*50)
    print(f"状态: 未检测到历史检查点，开始新训练")
    print(f"基础权重: {resume_path}")
    print(f"日志: {target_version_str}")
    print("="*50 + "\n")
    if args.version is not None:
        print(f"警告：您显式指定了日志版本 {target_version_str}\n")
# batch_size 为每张显卡的批大小。5090 显存极大，降低至 12 以预留验证阶段显存。
batch_size = 12 
logger_freq = 3000000000000 # 日志记录频率，此处设置得非常大以减少记录次数
learning_rate = 1e-5 # 学习率
only_mid_control = True # 标志位，可能用于控制模型中间层的行为
data_path = '/root/autodl-tmp/mvtecad/' # 数据集存放的根目录

# 初始化模型
# 首先在 CPU 上创建模型，PyTorch Lightning 会在训练开始时自动将其移动到配置好的 GPU 上
model = create_model('models/diad.yaml').cpu() # 根据 YAML 配置文件创建模型实例
# (可选) 开启编译优化，可提升速度。注：若采样逻辑包含复杂 Python 对象转换，可能会触发 Dynamo 错误。
# model = torch.compile(model) 

# 加载初始预训练权重 (仅当没有断点检查点时执行)
if not last_ckpt:
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    print(f"成功加载预训练权重: {resume_path}")

model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# 数据准备
# 创建 MVTec-AD 数据集的训练集和测试集（测试集在此用作验证集）
# 开启 load_to_ram=True 将完整数据集加载到内存中，利用 90GB 闲置内存加速训练
train_dataset = MVTecDataset('train', data_path, load_to_ram=True)
test_dataset = MVTecDataset('test', data_path, load_to_ram=True)

# 如果内存不足，请切换回以下代码（注释掉上面两行，取消下面两行的注释）
# train_dataset = MVTecDataset('train', data_path, load_to_ram=False)
# test_dataset = MVTecDataset('test', data_path, load_to_ram=False)
# 如需使用 VisA 数据集，可取消下面一行的注释
# train_dataset, test_dataset = VisaDataset('train',data_path), VisaDataset('test',data_path)

# # 使用小规模训练看看
# # ===== small dataset debug =====
# train_dataset.data = train_dataset.data[:100]
# test_dataset.data = test_dataset.data[:20]

# 创建 PyTorch 数据加载器（DataLoader）
# num_workers=8：在 load_to_ram 模式下，降低 worker 数量可减少 IPC 开销
# prefetch_factor=4：提高预取深度，确保 5090 始终有数据
# persistent_workers=True：保持进程，防止 Epoch 切换掉速
train_dataloader = DataLoader(
    train_dataset, 
    num_workers=8, 
    batch_size=batch_size, 
    shuffle=True, 
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)
test_dataloader = DataLoader(
    test_dataset, 
    num_workers=8, 
    batch_size=1, 
    shuffle=True, 
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

# 测试正常文本是否正确添加
# ===== DEBUG batch =====
# for batch in train_dataloader:
#     print("Batch text example:", batch["txt"][:4])
#     break

# 训练回调设置
# 保存验证集上准确率（val_acc）最高的模型权重到 ./val_ckpt/ 目录
ckpt_callback_val_loss = ModelCheckpoint(
    monitor='val_acc', 
    dirpath=ckpt_dir, 
    mode='max'
)
# 初始化自定义的图像日志记录器
logger = ImageLogger(batch_frequency=500)
# 初始化 TensorBoard 日志记录器
# 优先使用命令行指定的版本，或者是自动检测到的续训版本
tb_logger = TensorBoardLogger(
    save_dir="lightning_logs/", 
    name="", 
    version=latest_version if (last_ckpt or args.version is not None) else None
)

# 初始化 PyTorch Lightning 训练器（Trainer）
# accelerator="gpu", devices=1: 使用单 GPU 模式
# precision=16: 使用混合精度以节省显存
# callbacks: 包含图像记录和模型检查点保存
# accumulate_grad_batches=8: 梯度累加，由于 BS=1，累加 8 步后总等效 BS=8
# check_val_every_n_epoch=10: 每隔 10 个 epoch 进行一次验证
# trainer = pl.Trainer(
#     accelerator="gpu",
#     devices=1, 
#     precision=32, 
#     callbacks=[logger, ckpt_callback_val_loss], 
#     accumulate_grad_batches=8, 
#     max_epochs=150, 
#     check_val_every_n_epoch=10
# )
# trainer = pl.Trainer(
#         accelerator="gpu",
#         devices=1, 
#         precision="bf16-mixed", 
#         logger=tb_logger,
#         callbacks=[logger, ckpt_callback_val_loss], 
#         accumulate_grad_batches=1, 
#         max_epochs=200, 
#         check_val_every_n_epoch=10,
#         enable_progress_bar=True,
#         log_every_n_steps=50
#     )
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1, 
    precision="bf16-mixed", 
    logger=tb_logger,
    callbacks=[logger, ckpt_callback_val_loss], 
    accumulate_grad_batches=4, 
    max_epochs=200, 
    check_val_every_n_epoch=10,
    enable_progress_bar=True,
    log_every_n_steps=50
)
# trainer = pl.Trainer(
#         accelerator="gpu",
#         devices=1, 
#         precision="bf16-mixed", 
#         callbacks=[logger, ckpt_callback_val_loss], 
#         accumulate_grad_batches=8, 
#         max_epochs=2, 
#         check_val_every_n_epoch=1
#     )
# 开始执行训练流程
# 如果 last_ckpt 不为 None，则从断点继续，否则从 Epoch 0 开始
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader, ckpt_path=last_ckpt)
