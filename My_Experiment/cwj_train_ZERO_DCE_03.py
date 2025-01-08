import csv
from datetime import datetime
import shutil
import torch
import torch.optim
import os
import argparse
import time
import cwj_model
import cwj_Myloss
from torchvision import transforms
from cwj_Dataset_Low import LowDataset
from cwj_Dataset_Normal import NormalDataset
import pandas as pd
import matplotlib.pyplot as plt


'''
    保持ZERO_DCE的训练不变，在此基础上利用高质量图像数据
    低质量数据：通过DCE的方法训练
    高质量数据：将其转换为hsv色域，使得输入图像和输出图像的色彩饱和度保持一致，同时缩减正常照度图像数据数目

'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_save_folder(config):
    current_date_time = datetime.now().strftime("%y_%m_%d_%H_%M")
    save_folder = os.path.join(config.save_folder, "ZERO_DCE_" + current_date_time)
    os.makedirs(save_folder, exist_ok=True)

    return save_folder

def save_loss_records_to_csv(records, save_folder):
    """
    保存损失记录到 CSV 文件

    :param records: 损失记录列表，每项是一个字典
    :param filename: CSV 文件名
    """
    filename = os.path.join(save_folder , "loss_records.csv" )
    # 获取表头
    keys = records[0].keys()
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"Loss records saved to {filename}")

def plot_metrics_from_csv(save_folder):
    """
    读取CSV文件，绘制不同指标随轮次变化的折线图，并保存为图像文件。

    参数:
    - file_path: str, CSV文件的路径。
    - save_folder: str, 保存图像文件的文件夹路径。

    文件应包含以下格式的列:
    - epoch: 表示轮次。
    - 其他列: 不同的指标 (例如: Loss_TV, loss_spa 等)。
    """

    filename = os.path.join(save_folder, "loss_records.csv")
    data = pd.read_csv(filename)

    # 确保 'epoch' 列存在
    if 'epoch' not in data.columns:
        raise ValueError("CSV 文件中必须包含 'epoch' 列！")

    # 将 'epoch' 设为横轴
    epochs = data['epoch']

    # 创建图表
    plt.figure(figsize=(10, 6))
    for column in data.columns:
        if column != 'epoch':  # 排除 'epoch' 列
            plt.plot(epochs, data[column], label=column)

    # 设置图表标题和标签
    plt.title("Metrics over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # 创建保存路径
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "metrics_over_epochs.png")

    # 保存图像
    plt.savefig(save_path, dpi=300)
    print(f"图像已保存到: {save_path}")

def save_train_info(config, save_folder , low_csv_path , massage):
    """
    将训练配置参数保存到指定的文件夹下的 `train_info.txt` 文件中。

    参数:
    - config: argparse.Namespace, 包含训练配置参数。
    - save_folder: str, 保存 `train_info.txt` 的文件夹路径。
    """
    # 确保保存路径存在
    os.makedirs(save_folder, exist_ok=True)

    # 定义保存文件的完整路径
    save_path = os.path.join(save_folder, "train_info.txt")

    # 打开文件写入参数信息
    with open(save_path, 'w') as file:
        file.write(f"{massage}\n")
        file.write("Training Configuration Parameters:\n")
        file.write("=" * 40 + "\n")
        for key, value in vars(config).items():  # 将 config 转为字典遍历
            file.write(f"{key}: {value}\n")
        file.write("=" * 40 + "\n")

    print(f"训练配置已保存到: {save_path}")

    if os.path.exists(low_csv_path):
        # 确定目标路径
        csv_dest_path = os.path.join(save_folder, os.path.basename(low_csv_path))
        # 复制 CSV 文件
        shutil.copy2(low_csv_path, csv_dest_path)
        print(f"CSV 文件已复制到: {csv_dest_path}")
    else:
        print("low_csv_path 未提供或路径无效，跳过 CSV 文件复制。")


def check_for_nan_or_inf(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}!")

def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模型加载和初始化
    DCE_net = cwj_model.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    DCE_net.train()

    # 数据准备
    low_csv_path = r'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/ZERO_DCE_train.csv'
    normal_csv_path = r'/data/user/cwj_python310/Low_Light_Image/PaperAndCode/Zero_DCE_CVPR_2020/Zero-DCE_code/select_from_places365_1450.csv'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 转换为 Tensor
    ])
    low_dataset = LowDataset(low_csv_path, transform=transform)
    low_dataloader = torch.utils.data.DataLoader(low_dataset, batch_size=config.train_batch_size, shuffle=True)
    normal_dataset = NormalDataset(normal_csv_path, transform=transform)
    normal_dataloader = torch.utils.data.DataLoader(normal_dataset, batch_size=config.train_batch_size, shuffle=True)

    # 损失函数设定
    L_color = cwj_Myloss.L_color()
    L_spa = cwj_Myloss.L_spa(device=device)
    L_exp = cwj_Myloss.L_exp(16, 0.6 , device=device)
    L_TV = cwj_Myloss.L_TV()

    L_Saturation = cwj_Myloss.SaturationLoss()  # 计算两幅图之间饱和度通道的均方误差

    # 优化器设置
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 训练过程信息保存
    loss_list = []
    save_folder = get_save_folder(config)
    os.makedirs(save_folder, exist_ok=True)

    for epoch in range(config.num_epochs):
        start_time = time.time()  # 记录开始时间
        total_Loss_TV = 0
        total_loss_spa = 0
        total_loss_col = 0
        total_loss_exp = 0
        total_loss = 0

        total_loss_Saturation = 0

        for iteration , (img_normal , labels) in enumerate(normal_dataloader):
            img_normal = img_normal.to(device)
            _, enhanced_image, A = DCE_net(img_normal)

            # 计算损失函数
            loss_Saturation = config.Saturation_weight * L_Saturation(img_normal , enhanced_image)

            loss = loss_Saturation

            # 累加损失值
            total_loss_Saturation += loss_Saturation.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 训练数据保存
            if ((iteration + 1) % config.display_iter) == 0:
                print(f"epoch:{epoch + 1}/{config.num_epochs}: normal Loss at iteration", iteration + 1, ":", loss.item())

            if ((iteration + 1) % config.snapshot_iter) == 0:
                file_path = os.path.join(save_folder, "cwj_Epoch" + str(epoch) + '.pth')
                torch.save(DCE_net.state_dict(), file_path)

        for iteration, (img_lowlight, labels) in enumerate(low_dataloader):
            img_lowlight = img_lowlight.to(device)
            _, enhanced_image, A = DCE_net(img_lowlight)

            # 计算损失函数
            Loss_TV = config.TV_weight * L_TV(A)
            loss_spa = config.spa_weight * torch.mean(L_spa(img_lowlight, enhanced_image))
            loss_col = config.col_weight * torch.mean(L_color(enhanced_image))
            loss_exp = config.exp_weight * torch.mean(L_exp(enhanced_image))
            loss = Loss_TV + loss_spa + loss_col + loss_exp

            # 累加损失值
            total_Loss_TV += Loss_TV.item()
            total_loss_spa += loss_spa.item()
            total_loss_col += loss_col.item()
            total_loss_exp += loss_exp.item()
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 训练数据保存
            if ((iteration + 1) % config.display_iter) == 0:
                print(f"epoch:{epoch+1}/{config.num_epochs}: Low Loss at iteration", iteration + 1, ":", loss.item())

            # if ((iteration + 1) % config.snapshot_iter) == 0:
            #     file_path = os.path.join(save_folder,"cwj_Epoch" + str(epoch) + '.pth')
            #     torch.save(DCE_net.state_dict(), file_path)



        # 保存每轮次损失到记录列表
        loss_list.append({
            "epoch": epoch + 1,
            "Loss_TV": total_Loss_TV,
            "loss_spa": total_loss_spa,
            "loss_col": total_loss_col,
            "loss_exp": total_loss_exp,
            "total_loss": total_loss,
            "total_loss_Saturation" : total_loss_Saturation
        })
        end_time = time.time()  # 记录本轮结束时间
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"epoch:{epoch+1}/{config.num_epochs}: 耗时 {elapsed_time:.6f} 秒")

    save_loss_records_to_csv(loss_list , save_folder)
    plot_metrics_from_csv(save_folder)
    massage = "先用normal图像训练,再正常训练"
    save_train_info(config, save_folder, low_csv_path , massage)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data")
    parser.add_argument('--lr', type=float, default=0.0001)             # 学习率
    parser.add_argument('--weight_decay', type=float, default=0.00025)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=500)            # 训练轮次
    parser.add_argument('--train_batch_size', type=int, default=32)      # 训练 batch size
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)         # 每display_iter个批次输出本批次损失函数值
    parser.add_argument('--snapshot_iter', type=int, default=10)        # 每snapshot_iter轮保存模型权重
    parser.add_argument('--save_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")
    parser.add_argument('--TV_weight', type=int, default=200)
    parser.add_argument('--spa_weight', type=int, default=1)
    parser.add_argument('--col_weight', type=int, default=5)
    parser.add_argument('--exp_weight', type=int, default=10)
    parser.add_argument('--Saturation_weight', type=int, default=1)

    config = parser.parse_args()

    train(config)









