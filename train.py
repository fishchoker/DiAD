from share import *
import torch
import random
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.logger import ImageLogger
from sgn.model import create_model, load_state_dict
from visa_dataloader import VisaDataset
from pytorch_lightning.callbacks import ModelCheckpoint
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

setup_seed(1)
batch_size = 12 # 训练批大小（Batch Size）
logger_freq = 3000000000000 # 日志记录频率，此处设置得非常大以减少记录次数
learning_rate = 1e-5 # 学习率
only_mid_control = True # 标志位，可能用于控制模型中间层的行为
data_path = '/root/autodl-tmp/mvtecad/' # 数据集存放的根目录

# 初始化模型
# 首先在 CPU 上创建模型，PyTorch Lightning 会在训练开始时自动将其移动到配置好的 GPU 上
model = create_model('models/diad.yaml').cpu() # 根据 YAML 配置文件创建模型实例
# 加载预训练权重，strict=False 允许模型结构与权重文件有不完全匹配的部分
model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# 数据准备
# 创建 MVTec-AD 数据集的训练集和测试集（测试集在此用作验证集）
train_dataset, test_dataset = MVTecDataset('train',data_path), MVTecDataset('test',data_path)
# 如需使用 VisA 数据集，可取消下面一行的注释
# train_dataset, test_dataset = VisaDataset('train',data_path), VisaDataset('test',data_path)

# 使用小规模训练看看
# ===== small dataset debug =====
train_dataset.data = train_dataset.data[:100]
test_dataset.data = test_dataset.data[:20]

# 创建 PyTorch 数据加载器（DataLoader）
# num_workers=8 表示使用 8 个子进程并行加载数据
train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=1, shuffle=True)

# 测试正常文本是否正确添加
# ===== DEBUG batch =====
for batch in train_dataloader:
    print("Batch text example:", batch["txt"][:4])
    break

# 训练回调设置
# 保存验证集上准确率（val_acc）最高的模型权重到 ./val_ckpt/ 目录
ckpt_callback_val_loss = ModelCheckpoint(monitor='val_acc', dirpath='./val_ckpt/',mode='max')
# 初始化自定义的图像日志记录器
logger = ImageLogger(batch_frequency=logger_freq)

# 初始化 PyTorch Lightning 训练器（Trainer）
# gpus=1: 使用一个 GPU
# precision=32: 使用单精度浮点数
# callbacks: 包含图像记录和模型检查点保存
# accumulate_grad_batches=4: 梯度累加，每 4 个 batch 更新一次参数，相当于增大了 4 倍 batch size
# check_val_every_n_epoch=25: 每隔 25 个 epoch 进行一次验证集评估
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,ckpt_callback_val_loss], accumulate_grad_batches=4, check_val_every_n_epoch=25)
# 测试用的小规模trainer
trainer = pl.Trainer(
    gpus=1,
    precision=32,
    callbacks=[logger, ckpt_callback_val_loss],
    accumulate_grad_batches=1,
    max_epochs=2,
    limit_train_batches=50,
    limit_val_batches=10,
    check_val_every_n_epoch=1
)
# 开始执行训练流程
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)