import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import cwj_Calc_MUSIQ
import cwj_Calc_NIQE
import cwj_Calc_PSNR
import cwj_Calc_SSIM
import pandas as pd
import matplotlib.pyplot as plt
import pyiqa
import glob
import time


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])

def test_by_folder(input_dir, output_dir , model_dir , device):
    """
    在指定目录下使用模型生成同名增强结果
    """
    os.makedirs(output_dir, exist_ok=True)
    imgs = [img for img in os.listdir(input_dir) if is_image_file(img)]

    error_images = []

    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load(model_dir))
    DCE_net.to(device)
    DCE_net.eval()

    for img in imgs:
        img_path = os.path.join(input_dir, img)
        try:
            # print(f'Processing: {img_path}')

            # 加载并预处理图像
            data_lowlight = Image.open(img_path)
            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()
            data_lowlight = data_lowlight.permute(2, 0, 1)
            data_lowlight = data_lowlight.unsqueeze(0)
            data_lowlight = data_lowlight.to(device)

            # 前向传播得到增强的图像
            _, enhanced_image, _ = DCE_net(data_lowlight)

            # 保存结果
            result_path = os.path.join(output_dir, img)
            torchvision.utils.save_image(enhanced_image, result_path)

        except Exception as e:
            # 如果处理某张图像时出错，记录下图像的完整路径
            print(f"Error processing {img_path}: {str(e)}")
            error_images.append(img_path)

    return error_images


def save_musiq_niqe_to_csv(data, output_folder):
    # 创建文件路径
    csv_file_path = os.path.join(output_folder, "MUSIQ_NIQE.csv")
    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    # 保存为 CSV 文件
    df.to_csv(csv_file_path, index=False)
    print(f"MUSIQ_NIQE data saved to {csv_file_path}")

def plot_metrics_from_csv(output_folder):
    """
    读取CSV文件，分别绘制 MUSIQ 和 NIQE 随轮次变化的折线图，并保存为两张图像文件。

    参数:
    - output_folder: str, 保存CSV文件和图像文件的文件夹路径。

    文件应包含以下格式的列:
    - epoch: 表示轮次。
    - 其他列: 不同的指标 (例如: MUSIQ, NIQE)。
    """

    # CSV 文件路径
    filename = os.path.join(output_folder, "MUSIQ_NIQE.csv")
    data = pd.read_csv(filename)

    # 确保 'epoch' 列存在
    if 'epoch' not in data.columns:
        raise ValueError("CSV 文件中必须包含 'epoch' 列！")

    # 提取 epoch 和指标数据
    epochs = data['epoch']

    # 创建保存路径
    os.makedirs(output_folder, exist_ok=True)

    # 绘制 MUSIQ 图表
    if 'MUSIQ' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data['MUSIQ'], label='MUSIQ', color='orange')
        plt.axhline(y=58.2447, color='red', linestyle='--', label="MUSIQ Baseline (58.2447)")
        plt.title("MUSIQ over Epochs", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("MUSIQ Value", fontsize=14)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        musiq_save_path = os.path.join(output_folder, "MUSIQ_over_epochs.png")
        plt.savefig(musiq_save_path, dpi=300)
        plt.close()
        print(f"MUSIQ 图像已保存到: {musiq_save_path}")

    # 绘制 NIQE 图表
    if 'NIQE' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, data['NIQE'], label='NIQE', color='green')
        plt.axhline(y=4.2541, color='blue', linestyle='--', label="NIQE Baseline (4.2541)")
        plt.title("NIQE over Epochs", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("NIQE Value", fontsize=14)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        niqe_save_path = os.path.join(output_folder, "NIQE_over_epochs.png")
        plt.savefig(niqe_save_path, dpi=300)
        plt.close()
        print(f"NIQE 图像已保存到: {niqe_save_path}")

def delete_img(input_folder):
    '''
    保存每一轮的测试数据没有必要，所以在计算完NIQE和MUSIQ后删除增强后的图片
    :return:
    '''
    os.path.basename(input_folder)
    print(f"{os.path.basename(input_folder)}中图像已删除")
    if not os.path.isdir(input_folder):
        print(f"Error: The folder {input_folder} does not exist.")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 如果是文件且是图片类型，则删除
        if os.path.isfile(file_path) and is_image_file(filename):
            try:
                os.remove(file_path)  # 删除文件
                # print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


def test_on_small_datasets(device , epochs):
    '''
    在非成对数据集，DICM、NPE、LIME、NPE、VV数据集上进行评估
    '''
    test_dataset_folder_base = r"/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets"
    iqa_model_musiq = pyiqa.create_metric('musiq', device=device)
    iqa_model_niqe = pyiqa.create_metric('niqe', device=device)

    for root, dirs, files in os.walk(test_dataset_folder_base):
        for dir in dirs:
            test_dataset_folder = os.path.join(test_dataset_folder_base , dir)
            folder = r"ZERO_DCE_24_12_15_05_48"
            output_folder_base = rf"/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}/test_on_{dir}"

            MUSIQ_NIQE = []
            for epoch in range(epochs):
                model_dir = rf'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}/' + f"cwj_Epoch{epoch}.pth"
                output_folder = os.path.join(output_folder_base, os.path.basename(model_dir))
                # 使用模型生成增强结果
                test_by_folder(test_dataset_folder , output_folder , model_dir , device)
                print(output_folder)

                # 计算MUSIQ和NIQE
                _, average_MUSIQ = cwj_Calc_MUSIQ.Calculate_MUSIQ_for_folder(output_folder, output_folder, iqa_model_musiq,device)  # 每一轮模型的增强结果计算值放在增强结果文件夹中
                print(f"epoch：{epoch}/{epochs} average_MUSIQ：{average_MUSIQ}")
                _, average_NIQE = cwj_Calc_NIQE.Calculate_NIQE_for_folder(output_folder, output_folder, iqa_model_niqe,device)
                print(f"epoch：{epoch}/{epochs} average_NIQE：{average_NIQE}")
                MUSIQ_NIQE.append({
                    'epoch': epoch,
                    'MUSIQ': average_MUSIQ,
                    'NIQE': average_NIQE
                })
                delete_img(output_folder)  # 删除每一轮的测试数据
            save_musiq_niqe_to_csv(MUSIQ_NIQE, output_folder_base)
            plot_metrics_from_csv(output_folder_base)


def test_on_Paired_datasets(epochs, folder, device):
    test_paired_datasets_folder = [
        "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v1",
        # "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v2_real",
        "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v2_syn"
    ]

    output_folder_base = rf"/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}"

    # 如果epochs是整数，将其转换为range对象
    if isinstance(epochs, int):
        epochs = range(0, epochs)

    # 初始化存储结构，存储格式为：{epoch: {dataset_name: {PSNR, SSIM}}}
    results = {epoch: {} for epoch in epochs}

    for epoch in epochs:
        model_dir = os.path.join(output_folder_base, f"cwj_Epoch{epoch}.pth")

        for dataset_folder in test_paired_datasets_folder:
            dataset_name = os.path.basename(dataset_folder)
            low_dataset_folder = os.path.join(dataset_folder, "low")
            high_dataset_folder = os.path.join(dataset_folder, "high")
            enhanced_img_folder = os.path.join(output_folder_base, "test_on_" + dataset_name, f"cwj_Epoch{epoch}.pth")

            # 生成增强图像
            test_by_folder(low_dataset_folder, enhanced_img_folder, model_dir, device)

            # 增强图像与参考high亮度图像计算PSNR和SSIM
            nums, mean_PSNR = cwj_Calc_PSNR.calculate_psnr_for_images_in_folders(enhanced_img_folder, high_dataset_folder, enhanced_img_folder , device)
            _, mean_SSIM = cwj_Calc_SSIM.calculate_ssim_for_images_in_folders(enhanced_img_folder, high_dataset_folder, enhanced_img_folder , device)
            print(f"{low_dataset_folder} \n\tPSNR:{mean_PSNR}(nums:{nums}) \n\tSSIM:{mean_SSIM}(nums:{nums})")

            # 记录当前数据集的PSNR和SSIM
            results[epoch][dataset_name] = {"PSNR": mean_PSNR, "SSIM": mean_SSIM}

            # 删除增强结果
            delete_img(enhanced_img_folder)

    # 转换结果为表格格式
    formatted_results = []
    for epoch, datasets in results.items():
        row = {"epoch": epoch}
        for dataset_name, metrics in datasets.items():
            row[f"{dataset_name}_PSNR"] = metrics["PSNR"]
            row[f"{dataset_name}_SSIM"] = metrics["SSIM"]
        formatted_results.append(row)

    # 保存结果为CSV文件
    result_csv_path = os.path.join(output_folder_base, "PSNR_SSIM_results.csv")

    # 创建DataFrame并保存
    df = pd.DataFrame(formatted_results)
    df.to_csv(result_csv_path, index=False)

    print(f"PSNR and SSIM results saved to {result_csv_path}")



if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    folder = r"ZERO_DCE_24_12_15_05_48"
    # epochs = 500

    # 在DICM、VV这几个常用测试数据集上进行测试
    # test_on_small_datasets(device, epochs=500 , folder)
    # 在成对数据集LOL_v1
    # test_on_Paired_datasets(epochs , folder , device )

    ''' 绘制最好几轮增强图像'''
    epochs = [418 , 434 ]
    test_paired_datasets_folder = [
        "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v1",
        "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v2_real",
        "/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/LOL_v2_syn"
    ]
    output_folder_base = rf"/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}"

    for epoch in epochs:
        model_dir = os.path.join(output_folder_base, f"cwj_Epoch{epoch}.pth")

        for dataset_folder in test_paired_datasets_folder:
            dataset_name = os.path.basename(dataset_folder)
            low_dataset_folder = os.path.join(dataset_folder, "low")
            enhanced_img_folder = os.path.join(output_folder_base, "test_on_" + dataset_name, f"cwj_Epoch{epoch}.pth")

            # 生成增强图像
            test_by_folder(low_dataset_folder, enhanced_img_folder, model_dir, device)


    # output_folder_base = r'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/ZERO_DCE_24_12_15_05_48/test_on_DICM'
    # epochs = 500
    # for epoch in range(epochs):
    #     model_dir = r'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/ZERO_DCE_24_12_15_05_48/'+f"cwj_Epoch{epoch}.pth"
    #     input_folder = r'/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/DICM'
    #     output_folder = r'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/ZERO_DCE_24_12_15_05_48/test_on_DICM'
    #     output_folder = os.path.join(output_folder_base, os.path.basename(model_dir))
    #
    #     test_by_folder(input_folder, output_folder, model_dir , device)  # 使用对应模型生成增强结果
    #     print(output_folder)
    #     _,average_MUSIQ = cwj_Calc_MUSIQ.Calculate_MUSIQ_for_folder(output_folder , output_folder , iqa_model_musiq , device)   # 每一轮模型的增强结果计算值放在增强结果文件夹中
    #     print(f"epoch：{epoch}/{epochs} average_MUSIQ：{average_MUSIQ}")
    #     _,average_NIQE  = cwj_Calc_NIQE.Calculate_NIQE_for_folder(output_folder , output_folder , iqa_model_niqe , device)
    #     print(f"epoch：{epoch}/{epochs} average_NIQE：{average_NIQE}")
    #     MUSIQ_NIQE.append({
    #         'epoch': epoch,
    #         'MUSIQ': average_MUSIQ,
    #         'NIQE': average_NIQE
    #     })
    #     delete_img(output_folder)  # 删除每一轮的测试数据
    # save_musiq_niqe_to_csv(MUSIQ_NIQE, output_folder_base)
    # plot_metrics_from_csv(output_folder_base)

    # ''' 对特定轮次模型生成结果'''
    # folder = r"ZERO_DCE_24_12_15_05_48"
    # output_folder_base = rf'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}/test_on_VV'
    # epochs = [172 , 383 , 440 , 89 , 78]
    # for epoch in epochs:
    #     model_dir = rf'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}/' + f"cwj_Epoch{epoch}.pth"
    #     input_folder = r'/data/user/cwj_python310/Low_Light_Image/Datasets/small_test_datasets/VV'
    #     output_folder = rf'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/snapshots/{folder}/test_on_VV'
    #     output_folder = os.path.join(output_folder_base, os.path.basename(model_dir))
    #
    #     test_by_folder(input_folder, output_folder, model_dir, device)  # 使用对应模型生成增强结果
    #     print(output_folder)

