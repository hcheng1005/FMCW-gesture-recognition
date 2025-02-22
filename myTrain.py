import glob  
import numpy as np  
import torch  
from torch.utils.data import Dataset, DataLoader, random_split  
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
from torchvision.models import vgg11  
from torchvision.io import read_video  
from torchvision import transforms  
# !pip install pytorchvideo  
import pytorchvideo.transforms  
import matplotlib.pyplot as plt  
from tqdm import tqdm  
import re  
import random  
import cv2  
import gc  
import PIL  
# from google.colab.patches import cv2_imshow  
import h5py  

# 定义视频转换类  
class VideoTransform(object):  
    def __init__(self, size):  
        self.size = size  # 设置目标大小  

    def __call__(self, video):  
        h, w = self.size  
        L, C, H, W = video.size()  # 获取视频的维度  
        rescaled_video = torch.FloatTensor(L, C, h, w)  # 创建一个新的张量用于存储调整大小后的视频  

        # 计算视频的均值和标准差  
        vid_mean = video.mean()  
        vid_std = video.std()  

        # 定义转换操作  
        transform = transforms.Compose([  
            transforms.Resize(self.size, antialias=True),  # 调整大小  
            transforms.Normalize(0, 1),  # 归一化  
        ])  

        # 对每一帧进行转换  
        for l in range(L):  
            frame = video[l, :, :, :]  # 获取第l帧  
            frame = transform(frame)  # 应用转换  
            rescaled_video[l, :, :, :] = frame  # 存储转换后的帧  

        return rescaled_video  # 返回调整大小后的视频  


# 定义手势数据集类  
class GestureDataset(Dataset):  
    # 基于 https://github.com/fengxudi/mmWave-gesture-dataset  

    def __init__(self):  
        self.imgs_path = "/content/drive/MyDrive/Research project/mmWave-gesture-dataset/gesture_dataset/short_range_gesture/short_RangeDoppler/"  

        folder_list = glob.glob(self.imgs_path + "*")  # 获取所有文件夹路径  
        print(folder_list)  

        self.data = []  # 存储视频路径和标签  
        classes = set()  # 存储所有类别  

        # 遍历每个类别文件夹  
        for class_path in folder_list:  
            for video_path in glob.glob(class_path + "/*.mp4"):  # 获取每个视频文件  
                class_label = re.findall(r'short_RD_\d+_(\w+)_ $\d+$.mp4', video_path)[0]  # 提取类别标签  
                self.data.append([video_path, class_label])  # 存储视频路径和标签  
                classes.add(class_label)  # 添加类别到集合中  

        self.class_mapper = LabelEncoder()  # 创建标签编码器  
        print(classes)  
        self.class_mapper.fit([*classes])  # 拟合类别标签  

    def __len__(self):  
        return len(self.data)  # 返回数据集大小  

    def __getitem__(self, idx):  
        video_path, class_name = self.data[idx]  # 获取指定索引的视频路径和标签  

        # CNN需要的维度是(C, H, W)  
        video, _, __ = read_video(video_path, output_format='TCHW', pts_unit='sec')  # 读取视频  

        video = video.float()  # 转换为浮点型  
        video = VideoTransform((32, 32))(video)  # 应用视频转换  

        # 旋转视频  
        video = video.permute(0, 1, 3, 2)  # 调整维度顺序  

        class_id = self.class_mapper.transform([class_name])  # 转换类别标签为ID  
        class_id = torch.tensor(class_id)  # 转换为张量  

        return video, class_id  # 返回视频和标签  


# 定义第二个手势数据集类  
class GestureDataset2(Dataset):  
    # 基于 https://github.com/simonwsw/deep-soli  

    def __init__(self):  
        self.imgs_path = "/home/charles/Code/dataSet/SoliData/dsp/"  # 数据集路径  

        self.data = []  # 存储视频路径和标签  
        classes = set()  # 存储所有类别  

        video_paths = glob.glob(self.imgs_path + "*.h5")  # 获取所有h5文件  
        # print(video_paths)  

        # 遍历每个视频文件  
        for video_path in video_paths:  
            class_label = re.findall(r'(\d+)_\d+_\d+.h5', video_path)[0]  # 提取类别标签  
            self.data.append([video_path, class_label])  # 存储视频路径和标签  
            classes.add(class_label)  # 添加类别到集合中  

        self.class_mapper = LabelEncoder()  # 创建标签编码器  
        # print(classes)  
        self.class_mapper.fit([*classes])  # 拟合类别标签  

    def __len__(self):  
        return len(self.data)  # 返回数据集大小  

    def __getitem__(self, idx):  
        video_path, class_name = self.data[idx]  # 获取指定索引的视频路径和标签  

        outputs = []  # 存储每个通道的数据  

        with h5py.File(video_path, 'r') as f:  # 打开h5文件  
            for channel in range(3):  # 遍历每个通道  
                ch_data = f[f'ch{channel}']  # 获取通道数据  
                np_data = np.array(ch_data)  # 转换为numpy数组  
                np_data = np_data.reshape(-1, 32, 32)  # 重塑数据形状  
                tensor_data = torch.from_numpy(np_data)  # 转换为张量  
                outputs.append(tensor_data)  # 存储通道数据  

        video = torch.stack(outputs, dim=1)  # 将所有通道的数据堆叠在一起  
        video = video.float()  # 转换为浮点型  
        video = VideoTransform((32, 32))(video)  # 应用视频转换  

        class_id = self.class_mapper.transform([class_name])  # 转换类别标签为ID  
        class_id = torch.tensor(class_id)  # 转换为张量  

        return video, class_id  # 返回视频和标签  


# 定义特征提取器  
class FeatureExtractor(torch.nn.Module):  
    def __init__(self, in_features, out_features) -> None:  
        super().__init__()  
        # 定义卷积层和池化层  
        self.conv1 = torch.nn.Conv2d(in_features, 32, 3)  
        self.norm1 = torch.nn.BatchNorm2d(32)  
        self.pool1 = torch.nn.MaxPool2d(2)  

        self.conv2 = torch.nn.Conv2d(32, 64, 3)  
        self.norm2 = torch.nn.BatchNorm2d(64)  
        self.pool2 = torch.nn.MaxPool2d(2)  

        self.conv3 = torch.nn.Conv2d(64, 128, 3)  
        self.norm3 = torch.nn.BatchNorm2d(128)  
        self.pool3 = torch.nn.MaxPool2d(2)  

        self.conv4 = torch.nn.Conv2d(128, 256, 3)  
        self.norm4 = torch.nn.BatchNorm2d(256)  
        self.pool4 = torch.nn.MaxPool2d(2)  

        self.relu = torch.nn.ReLU()  # 激活函数  
        self.flatten = torch.nn.Flatten()  # 展平层  
        self.fc = torch.nn.Linear(512, out_features)  # 全连接层  
        self.dropout = torch.nn.Dropout(0.5)  # Dropout层  

    def forward(self, x):  
        # 定义前向传播过程  
        x = self.conv1(x)  
        x = self.norm1(x) 
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x) 
        x = self.norm2(x)  
        x = self.relu(x)  
        x = self.pool2(x)  

        x = self.conv3(x)  
        x = self.norm3(x)  
        x = self.relu(x)  
        x = self.pool3(x) 

        # x = self.conv4(x)  
        # x = self.norm4(x)  
        # x = self.relu(x)  
        # x = self.pool4(x)  

        x = self.flatten(x)  # 展平  
        x = self.fc(x)  # 全连接层  
        x = self.dropout(x)  # Dropout  

        return x  # 返回特征  


# 定义手势网络  
class GestureNet(torch.nn.Module):  
    def __init__(self, num_input_channels=4, num_cnn_features=256, num_rnn_hidden_size=256, num_classes=7) -> None:  
        super().__init__()  

        self.num_rnn_hidden_size = num_rnn_hidden_size  # RNN隐藏层大小  

        self.frame_model = FeatureExtractor(num_input_channels, num_cnn_features)  # 特征提取器  

        # self.frame_model = vgg11()  # 使用VGG11模型  
        # first_conv_layer = [torch.nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]  
        # first_conv_layer.extend(list(self.frame_model.features))  

        # self.frame_model.features = torch.nn.Sequential(*first_conv_layer)  
        # self.frame_model.classifier[6] = torch.nn.Linear(4096, num_cnn_features)  

        self.temporal_model = torch.nn.LSTM(input_size=num_cnn_features, hidden_size=num_rnn_hidden_size)  # LSTM层  

        self.fc1 = torch.nn.Linear(num_rnn_hidden_size, num_rnn_hidden_size // 2)  # 全连接层  
        self.relu = torch.nn.ReLU()  # 激活函数  
        self.fc2 = torch.nn.Linear(num_rnn_hidden_size // 2, num_classes)  # 输出层  

    def forward(self, x):  
        hidden = None  # 初始化隐藏状态  

        # 遍历每一帧  
        for frame in x:  
            features = self.frame_model(frame)  # 提取特征  
            features = torch.unsqueeze(features, 0)  # 增加维度  
            out, hidden = self.temporal_model(features, hidden)  # LSTM前向传播  

        out = torch.squeeze(out)  # 压缩维度  
        out = self.fc1(out)  # 全连接层  
        out = self.relu(out)  # 激活函数  
        out = self.fc2(out)  # 输出层  
        # out = torch.nn.Softmax(dim=1)(out)  # 计算Softmax  

        return out  # 返回输出  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU  

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

    cm = confusion_matrix(labels, predictions, normalize='all')  # 计算混淆矩阵  
    ConfusionMatrixDisplay(cm, display_labels=mapper.classes_).plot(xticks_rotation=45)  # 显示混淆矩阵  


# 保存模型的函数  
def save_model(model, num_classes, num_epochs, is_finetune=False, postfix=''):  
    core_path = "/home/charles/Code/FMCW-gesture-recognition/pth"  # 模型保存路径  
    path = f"{core_path}/trained_model_{'finetune' if is_finetune else 'pre-train'}_{num_classes}cl_{num_epochs}ep{postfix}.pt"  # 构建文件名  
    torch.save(model, path)  # 保存模型  


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

# 可视化样本的函数  
def visualize_sample(dataset, index=None):  
    if index is None:  
        index = random.randint(0, len(dataset))  # 如果没有指定索引，则随机选择一个索引  

    path, label = dataset.data[index]  # 获取指定索引的视频路径和标签  
    capture = cv2.VideoCapture(path)  # 打开视频文件  

    # 循环读取视频帧  
    while capture.isOpened():  
        ret, frame = capture.read()  # 读取一帧  

        if ret:  
            cv2.imshow(f'Action {label}', frame)  # 显示当前帧（注释掉的代码）  
            # cv2_imshow(frame)  # 使用cv2_imshow显示当前帧  
        else:  
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 如果没有读取到帧，则重置到视频开头  
            continue  # 继续循环  

        # 检测按键，如果按下'q'键，则退出循环  
        if cv2.waitKey(25) & 0xFF == ord('q'):  
            break  

    capture.release()  # 释放视频捕获对象  
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口  


# 自定义数据整理函数  
def collate_func(batch):  
    data, labels = zip(*batch)  # 解压批次数据和标签  

    labels = torch.tensor(labels)  # 将标签转换为张量  
    data = torch.nn.utils.rnn.pad_sequence(data)  # 对数据进行填充，使其具有相同的长度  

    return data, labels  # 返回填充后的数据和标签  


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
losses = list(run_training(train_loader, val_loader, dataset.class_mapper))  
# 可视化训练和验证的统计信息  
visualize_stats(losses)