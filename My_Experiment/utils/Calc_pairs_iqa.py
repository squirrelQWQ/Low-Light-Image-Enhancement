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


def calculate_pairs_iqa_for_images_in_folders(folder1, folder2, output_csv_folder, metrics, models, device):
    """
    计算多个图像质量指标（如 PSNR、SSIM、LPIPS）并生成CSV统计表格（包含img_name和各指标的数值）

    参数:
    folder1: str - 第一个图像文件夹的路径
    folder2: str - 第二个图像文件夹的路径
    output_csv_folder: str - 生成的CSV文件存放文件夹
    metrics: list - 要计算的指标列表 ['psnr', 'ssim', 'lpips']
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

    # 获取两个文件夹中的所有文件名
    filenames_folder1 = [f for f in os.listdir(folder1) if is_image_file(f)]
    filenames_folder2 = [f for f in os.listdir(folder2) if is_image_file(f)]

    # 确保两个文件夹有相同的文件
    common_filenames = set(filenames_folder1).intersection(filenames_folder2)

    # 遍历所有共同的图像文件
    for filename in common_filenames:
        img_path1 = os.path.join(folder1, filename)
        img_path2 = os.path.join(folder2, filename)

        # 使用PIL读取图像
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')

        # 将PIL图像转换为numpy数组
        img1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # 计算各个指标并存储结果
        img_names.append(filename)
        for metric in metrics:
            metric_value = models[metric](img1_tensor, img2_tensor)
            # 将计算结果从 Tensor 转换为标量值，并存入 metrics_values
            metrics_values[metric].append(
                metric_value.cpu().item() if isinstance(metric_value, torch.Tensor) else metric_value)

    # 计算所有指标的均值
    mean_metrics = {metric: np.mean(values) for metric, values in metrics_values.items()}

    # 将结果保存到 CSV 文件
    output_csv_path = os.path.join(output_csv_folder, 'pairs_iqa_PSNR_SSIM_LPIPS.csv')
    df = pd.DataFrame({'img_name': img_names, **metrics_values})
    df.to_csv(output_csv_path, index=False)

    # 返回 PSNR 数量和每个指标的均值
    return len(metrics_values['psnr']), mean_metrics



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    folder1 = r'/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v1/low'
    folder2 = r'/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v1/high'
    output_csv_folder = r'/data/user/cwj_python310/Low_Light_Image/Low_Light_Image_Enhancement/utils'

    # 需要计算的指标
    metrics = ['psnr', 'ssim', 'lpips']

    # 初始化 pyIQA 库的模型
    models = {}
    for metric in metrics:
        models[metric] = pyiqa.create_metric(metric, device=device)

    # 计算图像质量指标
    nums, mean_metrics = calculate_pairs_iqa_for_images_in_folders(folder1, folder2, output_csv_folder, metrics, models, device)

    # 输出结果
    print(f"Metrics calculated for {nums} images. Mean values: {mean_metrics}")