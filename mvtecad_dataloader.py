# 导入必要的库
import json
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# MVTec 数据集通用的 ImageNet 均值和标准差，用于归一化
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(size):
    """
    对训练/测试图像进行的预处理操作
    :param size: 目标图像尺寸 (size x size)
    :return: 组合后的 transforms 序列
    """
    datatrans =  transforms.Compose([
    transforms.Resize((size, size)), # 缩放图像
    transforms.ToTensor(),           # 转换为 Tensor 格式
    transforms.CenterCrop(size),     # 中心裁剪
    transforms.Normalize(mean=mean_train,
                         std=std_train)]) # 归一化
    return datatrans

def gt_transforms(size):
    """
    对 Ground Truth (Mask) 进行的预处理操作
    :param size: 目标 Mask 尺寸
    :return: 组合后的 transforms 序列
    """
    gttrans =  transforms.Compose([
    transforms.Resize((size, size)),
    transforms.CenterCrop(size),
    transforms.ToTensor()])
    return gttrans


class MVTecDataset(Dataset):
    """
    MVTec-AD 数据集加载类
    """
    def __init__(self, type, root, load_to_ram=False):
        """
        初始化数据集
        :param type: 'train' 加载训练集，否则加载测试集
        :param root: 数据集的根目录路径
        :param load_to_ram: 是否预加载整个数据集到内存中
        """
        self.data = []
        # 从 JSON 文件加载数据索引信息
        if type == 'train':
            with open('./training/MVTec-AD/train.json', 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            with open('./training/MVTec-AD/test.json', 'rt') as f:
                for line in f:
                    self.data.append(json.loads(line))
        
        # 15个类别的物理名称清单 (此顺序为系统唯一、不可更改的物理索引参考)
        self.CLASS_NAMES = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
            'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
        
        # 建立类别名称到物理索引 (0-14) 的唯一映射，消除任何 Shuffle 或字典排序带来的漂移
        self.label_to_idx = {name: str(idx) for idx, name in enumerate(self.CLASS_NAMES)}
        
        self.image_size = (256, 256) # 默认图像尺寸
        self.root = root

        # 15个类别的 Top 3 高分提示词 (根据 CLIP 评估结果更新)
        self.class_prompts = {
            'bottle': [
                "a photo of the bottle without damage for anomaly detection.",
                "a close-up photo of a bottle.",
                "a close-up photo of a bottle without damage."
            ],
            'cable': [
                "a photo of a unblemished cable for visual inspection.",
                "a close-up photo of the cable without damage.",
                "a close-up photo of a unblemished cable"
            ],
            'capsule': [
                "a photo of a small capsule without flaw.",
                "a photo of the small capsule without flaw.",
                "a close-up photo of a capsule without flaw."
            ],
            'carpet': [
                "a photo of the carpet without flaw for anomaly detection.",
                "a cropped photo of the carpet without defect.",
                "a photo of the carpet for anomaly detection"
            ],
            'grid': [
                "a close-up photo of a grid without damage.",
                "a close-up photo of a grid without defect.",
                "a cropped photo of a grid without defect."
            ],
            'hazelnut': [
                "a cropped photo of a hazelnut.",
                "a cropped photo of a hazelnut without damage.",
                "a photo of a hazelnut without damage for anomaly detection."
            ],
            'leather': [
                "a cropped photo of a leather.",
                "a cropped photo of a leather without damage.",
                "a close-up photo of a leather without damage."
            ],
            'metal_nut': [
                "a photo of a metal nut without defect for visual inspection.",
                "a close-up photo of a metal nut without defect.",
                "a photo of a unblemished metal nut for visual inspection."
            ],
            'pill': [
                "a photo of a unblemished pill for visual inspection.",
                "a photo of a pill without flaw for visual inspection.",
                "a photo of the pill without flaw for visual inspection."
            ],
            'screw': [
                "a photo of a small screw.",
                "a photo of a small unblemished screw.",
                "a photo of a small screw without damage."
            ],
            'tile': [
                "a close-up photo of a tile without damage.",
                "a close-up photo of a tile without defect.",
                "a photo of the tile without damage."
            ],
            'toothbrush': [
                "a photo of a unblemished toothbrush for visual inspection.",
                "a photo of the toothbrush without defect.",
                "a close-up photo of the toothbrush without defect."
            ],
            'transistor': [
                "a photo of a transistor without damage for anomaly detection.",
                "a photo of the transistor without damage for anomaly detection.",
                "a photo of the small transistor without damage."
            ],
            'wood': [
                "a cropped photo of the wood surface without defect.",
                "a cropped photo of a wood surface without defect.",
                "a close-up photo of the wood surface without defect."
            ],
            'zipper': [
                "a photo of the unblemished zipper for anomaly detection.",
                "a cropped photo of a zipper without flaw.",
                "a cropped photo of the zipper without flaw."
            ]
        }

        # 加载到内存逻辑
        self.load_to_ram = load_to_ram
        self.samples = []
        if self.load_to_ram:
            print(f"Loading {type} dataset to RAM (Total: {len(self.data)} samples)...")
            for i in tqdm(range(len(self.data))):
                self.samples.append(self.get_sample(i))
            print(f"Successfully preloaded {type} dataset.")

    def __len__(self):
        """
        返回数据集样本总数
        """
        return len(self.data)

    def get_sample(self, idx):
        """
        内部方法：从磁盘读取并处理单个样本
        """
        item = self.data[idx]
        source_filename = item['filename'] # 源文件名
        target_filename = item['filename'] # 目标文件名（在重建任务中通常相同）
        label = item["label"]
        
        # 加载 Mask
        if item.get("maskname", None):
            # 如果有 maskname，从磁盘加载异常掩码
            mask = cv2.imread( self.root + item['maskname'], cv2.IMREAD_GRAYSCALE)
        else:
            # 如果没有 maskname，根据 label 生成全黑或全白掩码
            if label == 0:  # good: 正常样本，生成全黑 Mask
                mask = np.zeros(self.image_size).astype(np.uint8)
            elif label == 1:  # defective: 异常样本（但未提供具体掩码时），生成全白 Mask
                mask = (np.ones(self.image_size)).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        #prompt = "" # 提示词，默认为空
        prompt = self.class_prompts[item["clsname"]] # 获取对应类别的 Top 3 提示词列表


        # 读取图像并转换颜色空间 (OpenCV BGR -> RGB)
        source = cv2.imread(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image 对象以便进行 transforms
        source = Image.fromarray(source, "RGB")
        target = Image.fromarray(target, "RGB")
        mask = Image.fromarray(mask, "L")
        
        # 统一缩放尺寸
        transform_fn = transforms.Resize(self.image_size)
        source = transform_fn(source)
        target = transform_fn(target)
        mask = transform_fn(mask)
        
        # 转换为 Tensor
        source = transforms.ToTensor()(source)
        target = transforms.ToTensor()(target)
        mask = transforms.ToTensor()(mask)
        
        # 对图像进行归一化
        normalize_fn = transforms.Normalize(mean=mean_train, std=std_train)
        source = normalize_fn(source)
        target = normalize_fn(target)
        
        clsname = item["clsname"] # 获取类别名称
        image_idx = self.label_to_idx[clsname] # 获取类别索引

        return {
            'jpg': source, 
            'txt': prompt, 
            'hint': target, 
            'mask': mask, 
            'clsname': clsname, 
            'image_idx': image_idx, 
            'filename': source_filename, 
            'label': label
        }

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        """
        if self.load_to_ram:
            return self.samples[idx]
        return self.get_sample(idx)
