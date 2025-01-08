import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import pyiqa


def is_image_file(filename):
    """
    检查文件是否为图片文件（根据文件扩展名）
    """
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])


def calculate_nr_iqa_for_images_in_folders(folder1, output_csv_folder, metrics, models, device):
    """
    计算无参考图像质量评价指标（如 MUSIQ、NIQE、BRISQUE、PIQE）并生成CSV统计表格（包含img_name和各指标的数值）

    参数:
    folder1: str - 图像文件夹的路径
    output_csv_folder: str - 生成的CSV文件存放文件夹
    metrics: list - 要计算的指标列表 ['musiq', 'niqe', 'brisque', 'piqe']
    models: dict - 初始化的 pyIQA 库模型，字典形式存储各个指标的模型
    device: torch.device - 设备（CPU 或 GPU）

    返回值:
    tuple:
        - nums：计算指标图像的数量
        - mean_metrics：包含各个指标均值的字典
    """
    # 存储每个指标的值
    metrics_values = {metric: [] for metric in metrics}
    img_names = []

    # 获取文件夹中的所有文件名
    filenames_folder1 = [f for f in os.listdir(folder1) if is_image_file(f)]

    # 遍历所有图像文件
    for filename in filenames_folder1:
        img_path = os.path.join(folder1, filename)

        # 使用PIL读取图像
        img = Image.open(img_path).convert('RGB')

        # 将PIL图像转换为numpy数组
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)

        # 计算各个指标并存储结果
        img_names.append(filename)
        for metric in metrics:
            metric_value = models[metric](img_tensor)
            # 将计算结果从 Tensor 转换为标量值，并存入 metrics_values
            metrics_values[metric].append(metric_value.cpu().item() if isinstance(metric_value, torch.Tensor) else metric_value)

    # 计算所有指标的均值
    mean_metrics = {metric: np.mean(values) for metric, values in metrics_values.items()}

    # 将结果保存到 CSV 文件
    output_csv_path = os.path.join(output_csv_folder, 'unpairs_iqa_MUSIQ_NIQE_BRISQUE_PIQE.csv')
    df = pd.DataFrame({'img_name': img_names, **metrics_values})
    df.to_csv(output_csv_path, index=False)

    # 返回计算指标图像的数量和每个指标的均值
    return len(metrics_values[metrics[0]]), mean_metrics


if __name__ == '__main__':
    # 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    folder1 = r'/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v2_real/low'  # 图像文件夹路径
    output_csv_folder = r'/data/user/cwj_python310/Low_Light_Image/Low_Light_Image_Enhancement/utils'  # 输出CSV文件夹路径

    # 初始化 pyIQA 库的模型
    models = {
        'musiq': pyiqa.create_metric('musiq', device=device),
        'niqe': pyiqa.create_metric('niqe', device=device),
        'brisque': pyiqa.create_metric('brisque', device=device),
        'piqe': pyiqa.create_metric('piqe', device=device),
    }

    # 指定计算的指标
    metrics = ['musiq', 'niqe', 'brisque', 'piqe']

    # 计算指标
    nums, mean_metrics = calculate_nr_iqa_for_images_in_folders(folder1, output_csv_folder, metrics, models, device)

    # 输出结果
    print(f"Metrics calculated for {nums} images. Mean values:")
    for metric, value in mean_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
