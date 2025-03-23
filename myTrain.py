import glob  
import numpy as np  
import torch  
import random
from torch.utils.data import Dataset, DataLoader, random_split  
import pytorchvideo.transforms  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
import matplotlib.pyplot as plt  
from tqdm import tqdm  
import re  
import random  
# import cv2  
import gc  
import PIL  
import pickle  

from model import GestureDataset2, GestureNet



# 训练模型的函数  
def train_model(model, loader, optimizer, scheduler, loss_func, device):  
    losses = []  # 存储损失  
    model.train()  # 设置模型为训练模式  

    # 遍历数据加载器  
    for _, (frames, label) in enumerate(tqdm(loader)):  
        optimizer.zero_grad()  # 清零梯度  

        frames = frames.to(device)  # 将数据移动到设备  
        label = label.to(device)  # 将标签移动到设备  

        y_pred = model(frames)  # 前向传播  
        loss = loss_func(y_pred, label)  # 计算损失  
        losses.append(loss.item())  # 存储损失  

        loss.backward()  # 反向传播  
        optimizer.step()  # 更新参数  

    scheduler.step()  # 更新学习率  

    return np.mean(losses)  # 返回平均损失  


# 验证模型的函数  
def validate_model(model, loader, loss_func, device):  
    losses = []  # 存储损失  
    acc = 0.0  # 初始化准确率  
    total = 0  # 初始化总样本数  
    model.eval()  # 设置模型为评估模式  

    labels = []  # 存储真实标签  
    predictions = []  # 存储预测标签  

    with torch.no_grad():  # 不计算梯度  
        for frames, label in tqdm(loader):  
            frames = frames.to(device)  # 将数据移动到设备  
            label = label.to(device)  # 将标签移动到设备  

            y_pred = model(frames)  # 前向传播  

            loss = loss_func(y_pred, label)  # 计算损失  
            losses.append(loss.item())  # 存储损失  

            acc += sum(label == y_pred.argmax(dim=1)).item()  # 计算准确率  
            total += len(label)  # 更新总样本数  

            labels.extend(label.detach().cpu().numpy())  # 存储真实标签  
            predictions.extend(y_pred.argmax(dim=1).detach().cpu().numpy())  # 存储预测标签  

    return np.mean(losses), acc / total, labels, predictions  # 返回平均损失、准确率、真实标签和预测标签  


# 运行训练的主函数  
def run_training(train_loader, val_loader, mapper, num_epochs=2, num_classes=12, model_postfix=''):  
    model = GestureNet(num_input_channels=3, num_classes=num_classes)  # 初始化模型  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)  # 学习率调度器  
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数  

    model.to(device)  # 将模型移动到设备  

    best_state = model.state_dict()  # 保存最佳模型状态  
    best_loss = 1000000  # 初始化最佳损失  
    best_epoch = 0  # 初始化最佳轮次  

    for index in tqdm(range(num_epochs)):  
        train_loss = train_model(model, train_loader, optimizer, scheduler, loss_func, device)  # 训练模型  
        val_loss, accuracy, labels, predictions = validate_model(model, val_loader, loss_func, device)  # 验证模型  

        # 如果验证损失更好，则更新最佳状态  
        if val_loss < best_loss:  
            best_loss = val_loss  
            best_state = model.state_dict()  
            best_epoch = index  

        print('\n', train_loss, val_loss, accuracy)  # 打印损失和准确率  
        yield (train_loss, val_loss, accuracy)  # 生成训练和验证损失  

    model.load_state_dict(best_state)  # 加载最佳模型状态  
    save_model(model, num_classes, num_epochs, is_finetune=False, postfix=model_postfix)  # 保存模型  
    loss, acc, labels, predictions = validate_model(model, val_loader, loss_func, device)  # 再次验证模型  

    print(f'Best epoch: {best_epoch}. Final loss: {loss}, acc: {acc}')  # 打印最佳轮次和最终损失  

    # cm = confusion_matrix(labels, predictions, normalize='all')  # 计算混淆矩阵  
    # ConfusionMatrixDisplay(cm, display_labels=mapper.classes_).plot(xticks_rotation=45)  # 显示混淆矩阵  


# 保存模型的函数  
def save_model(model, num_classes, num_epochs, is_finetune=False, postfix=''):  
    core_path = "/home/charles/Code/FMCW-gesture-recognition/pth"  # 模型保存路径  
    path = f"{core_path}/trained_model_{'finetune' if is_finetune else 'pre-train'}_{num_classes}cl_{num_epochs}ep{postfix}.pt"  # 构建文件名  
    torch.save({'model_state_dict': model.state_dict()},  path)  # 保存模型  


# 可视化训练和验证统计信息的函数  
def visualize_stats(losses):  
    _, axs = plt.subplots(3, figsize=(10, 18))  # 创建子图  
    indices = np.arange(1, len(losses) + 1, 1)  # 创建索引  

    axs[0].plot(indices, [train_loss for train_loss, _, _ in losses])  # 绘制训练损失  
    axs[0].set_title('Train loss')  # 设置标题  
    axs[0].set_xlabel('Epoch')  # 设置x轴标签  
    axs[0].set_ylabel('Loss')  # 设置y轴标签  
    axs[0].grid(axis='y', linewidth=0.4)  # 添加网格  
    axs[0].set(frame_on=False)  # 关闭框架  

    axs[1].plot(indices, [val_loss for _, val_loss, _ in losses])  # 绘制验证损失  
    axs[1].set_title('Validation loss')  # 设置标题  
    axs[1].set_xlabel('Epoch')  # 设置x轴标签  
    axs[1].set_ylabel('Loss')  # 设置y轴标签  
    axs[1].grid(axis='y', linewidth=0.4)  # 添加网格  
    axs[1].set(frame_on=False)  # 关闭框架  

    axs[2].plot(indices, [acc * 100 for _, _, acc in losses])  # 绘制验证准确率  
    axs[2].set_title('Validation accuracy')  # 设置标题  
    axs[2].set_xlabel('Epoch')  # 设置x轴标签  
    axs[2].set_ylabel('Accuracy (%)')  # 设置y轴标签  
    axs[2].grid(axis='y', linewidth=0.4)  # 添加网格  
    axs[2].set(frame_on=False)  # 关闭框架  

    plt.show()  # 显示图形  

# # 可视化样本的函数  
# def visualize_sample(dataset, index=None):  
#     if index is None:  
#         index = random.randint(0, len(dataset))  # 如果没有指定索引，则随机选择一个索引  

#     path, label = dataset.data[index]  # 获取指定索引的视频路径和标签  
#     capture = cv2.VideoCapture(path)  # 打开视频文件  

#     # 循环读取视频帧  
#     while capture.isOpened():  
#         ret, frame = capture.read()  # 读取一帧  

#         if ret:  
#             cv2.imshow(f'Action {label}', frame)  # 显示当前帧（注释掉的代码）  
#             # cv2_imshow(frame)  # 使用cv2_imshow显示当前帧  
#         else:  
#             capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 如果没有读取到帧，则重置到视频开头  
#             continue  # 继续循环  

#         # 检测按键，如果按下'q'键，则退出循环  
#         if cv2.waitKey(25) & 0xFF == ord('q'):  
#             break  

#     capture.release()  # 释放视频捕获对象  
#     cv2.destroyAllWindows()  # 关闭所有OpenCV窗口  


# 自定义数据整理函数  
def collate_func(batch):  
    data, labels = zip(*batch)  # 解压批次数据和标签  

    labels = torch.tensor(labels)  # 将标签转换为张量  
    data = torch.nn.utils.rnn.pad_sequence(data)  # 对数据进行填充，使其具有相同的长度  

    return data, labels  # 返回填充后的数据和标签  






if __name__ == "__main__":   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU  

    # 确保可重复性  
    torch.manual_seed(123)  # 设置PyTorch的随机种子  
    np.random.seed(123)  # 设置NumPy的随机种子  
    random.seed(123)  # 设置Python内置random模块的随机种子  

    # 初始化手势数据集  
    dataset = GestureDataset2()  
    # 将数据集随机分为训练集和验证集，比例为70%和30%  
    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])  

    # 创建数据加载器，设置批量大小和是否打乱数据  
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_func)  
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_func)  

    # 可视化样本（注释掉的代码）  
    # visualize_sample(dataset)  

    # 运行训练过程，并将损失记录为列表  
    losses = list(run_training(train_loader, val_loader, dataset.class_mapper, num_epochs=32))  

    # 使用二进制写入方式保存训练数据便于后续查看
    with open("losses_data.pkl", "wb") as f:  
        pickle.dump(losses, f)  

    # # 可视化训练和验证的统计信息  
    # visualize_stats(losses)