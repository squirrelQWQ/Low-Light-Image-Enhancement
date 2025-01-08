import os
from random import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class NormalDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path, header=None)
        self.images = [(row[0], 1) for _, row in df.iterrows()] # 所有图像的标签都设为 1 表示为正常照度图像
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
        if self.transform:
            image = self.transform(image)

        return image, label


def test():
    low_csv_path = '../low_img_720P.csv'

    # 定义图像的预处理操作
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 将图像调整为 256x256
        transforms.ToTensor(),  # 转换为 Tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    # 创建数据集实例
    low_dataset = LowDataset(low_csv_path, transform=transform)
    print(len(low_dataset))

    # 使用 DataLoader 加载数据集
    low_dataloader = DataLoader(low_dataset, batch_size=4, shuffle=True)

    # 取一个 batch 进行测试
    for i, (images, labels) in enumerate(low_dataloader):
        print(f"Batch {i + 1}")
        print(f"Images shape: {images.shape}")  # 打印图像的 shape
        print(f"Labels: {labels}")  # 打印标签

        # # 为了测试，只显示一个 batch
        # break



# 示例用法
if __name__ == "__main__":
    test()
