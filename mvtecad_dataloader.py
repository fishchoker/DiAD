# 导入必要的库
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
    def __init__(self, type, root):
        """
        初始化数据集
        :param type: 'train' 加载训练集，否则加载测试集
        :param root: 数据集的根目录路径
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
        
        # 15个类别的名称到索引的映射
        self.label_to_idx = {'bottle': '0', 'cable': '1', 'capsule': '2', 'carpet': '3', 'grid': '4', 'hazelnut': '5',
                             'leather': '6', 'metal_nut': '7', 'pill': '8', 'screw': '9', 'tile': '10',
                             'toothbrush': '11', 'transistor': '12', 'wood': '13', 'zipper': '14'}
        self.image_size = (256, 256) # 默认图像尺寸
        self.root = root

        #补上各个类别对应的正常描述
        self.class_prompts = {
            'bottle': 'a photo of a normal bottle without defect',
            'cable': 'a photo of a normal cable without defect',
            'capsule': 'a photo of a normal capsule without defect',
            'carpet': 'a photo of a normal carpet without defect',
            'grid': 'a photo of a normal grid without defect',
            'hazelnut': 'a photo of a normal hazelnut without defect',
            'leather': 'a photo of a normal leather without defect',
            'metal_nut': 'a photo of a normal metal nut without defect',
            'pill': 'a photo of a normal pill without defect',
            'screw': 'a photo of a normal screw without defect',
            'tile': 'a photo of a normal tile without defect',
            'toothbrush': 'a photo of a normal toothbrush without defect',
            'transistor': 'a photo of a normal transistor without defect',
            'wood': 'a photo of a normal wood surface without defect',
            'zipper': 'a photo of a normal zipper without defect'
        }

    def __len__(self):
        """
        返回数据集样本总数
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        :param idx: 样本索引
        :return: 包含图像、掩码、提示词、文件名、类别名、标签的字典
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
        prompt = self.class_prompts[item["clsname"]] # 获取对应类别正常描述


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

        # 返回符合 DiAD 训练架构的字典格式
        return dict(
            jpg=target,         # 重建目标图
            txt=prompt,         # 文本提示（修改后应为正常语义先验
            hint=source,        # 条件控制图（原图）
            mask=mask,          # 异常区域掩码
            filename=source_filename, 
            clsname=clsname, 
            label=int(image_idx) # 类别索引（用于多类别模型控制）
        )

