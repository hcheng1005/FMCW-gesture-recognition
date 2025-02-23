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
# from google.colab.patches import cv2_imshow  
import h5py  
import re  

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


class FinetuneGestureNet(GestureNet):
    def __init__(self, num_classes,
                 weights_path="/content/drive/MyDrive/research_project_models/trained_model_25ep.pt"):
        super().__init__(num_input_channels=3, num_classes=12)

        self.load_state_dict(torch.load(weights_path).state_dict())
        # self.fc2 = torch.nn.Linear(self.num_rnn_hidden_size // 2, num_classes)
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(self.num_rnn_hidden_size // 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
