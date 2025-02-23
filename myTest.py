import glob  
import numpy as np  
import torch  
import random
from torch.utils.data import Dataset, DataLoader, random_split  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
from tqdm import tqdm  
import matplotlib.pyplot as plt  
from model import GestureDataset2, GestureNet, FeatureExtractor


# 自定义数据整理函数  
def collate_func(batch):  
    data, labels = zip(*batch)  # 解压批次数据和标签  

    labels = torch.tensor(labels)  # 将标签转换为张量  
    data = torch.nn.utils.rnn.pad_sequence(data)  # 对数据进行填充，使其具有相同的长度  

    return data, labels  # 返回填充后的数据和标签  


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
    
    
    

# 设置设备为GPU或CPU  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

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


# 构造模型
model = GestureNet(num_input_channels=3, num_classes=12)

# 加载权重文件
checkpoint = torch.load("/home/charles/Code/FMCW-gesture-recognition/pth/trained_model_pre-train_12cl_100ep.pt", 
                        map_location=torch.device("cuda"),
                        weights_only=True)  

model.load_state_dict(checkpoint['model_state_dict'])  
model.eval()
model.to(device)  # 将模型移动到设备  


loss_func = torch.nn.CrossEntropyLoss()  # 损失函数  

val_loss, accuracy, labels, predictions = validate_model(model, val_loader, loss_func, device)  # 验证模型  
print('\n', val_loss, accuracy)  # 打印损失和准确率 

cm = confusion_matrix(labels, predictions, normalize='all')  # 计算混淆矩阵  
disp = ConfusionMatrixDisplay(cm, display_labels=dataset.class_mapper.classes_).plot(xticks_rotation=45)  # 显示混淆矩阵 
disp.plot()
plt.show()