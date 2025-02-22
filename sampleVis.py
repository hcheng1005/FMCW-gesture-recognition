import os  
import numpy as np  
from matplotlib import pyplot as plt  
import h5py  
import imageio  
from tqdm import tqdm  # 导入tqdm库  

# 设置要遍历的文件夹路径和GIF存储路径  
folder_path = '/home/charles/Code/dataSet/SoliData/dsp/'  
gif_output_path = '/home/charles/Code/dataSet/SoliData/GIF/'  

# 确保GIF存储路径存在  
os.makedirs(gif_output_path, exist_ok=True)  

# 获取文件夹中的所有H5文件，并按文件名排序  
h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])  

# 遍历每个H5文件  
for file_name in tqdm(h5_files, desc="Processing H5 files"):  
    h5_file_path = os.path.join(folder_path, file_name)  
    use_channel = 0  # 选择通道0的数据  

    # 读取H5文件  
    with h5py.File(h5_file_path, 'r') as f:  
        data = f['ch{}'.format(use_channel)][()]  
        label = f['label'][()]  

    data = data.reshape(data.shape[0], 32, 32)  # 重塑数据为32x32的图像  

    # 用于保存每一帧的列表  
    frames = []  

    for i in range(data.shape[0]):  
        # 创建图形并绘制图像  
        fig, ax = plt.subplots()  
        # ax.imshow(data[i], cmap='gray')  # 显示图像  
        ax.imshow(data[i])  # 保存实际值而非灰度图 
        plt.axis('off')  # 关闭坐标轴  

        # 保存当前帧为图像  
        plt.savefig('temp_frame.png', bbox_inches='tight', pad_inches=0)  # 保存当前帧为临时文件  
        plt.close(fig)  # 关闭图形窗口以释放资源  

        frames.append(imageio.v2.imread('temp_frame.png'))  # 读取并添加到帧列表  

    # 生成GIF文件名  
    gif_file_name = os.path.splitext(file_name)[0] + '.gif'  
    gif_file_path = os.path.join(gif_output_path, gif_file_name)  

    # 保存为GIF  
    imageio.mimsave(gif_file_path, frames, duration=0.05)  # duration为每帧持续时间  

print("All GIFs have been created successfully.")