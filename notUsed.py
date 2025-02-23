import numpy as np   
import matplotlib.pyplot as plt 



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

loss_ = list([(2.4028271880031618, 2.3443697083454866, 0.14), (2.1999863947718596, 2.6778667007501307, 0.13090909090909092), (1.6593918721537946, 1.6151034184373343, 0.3484848484848485), (1.2975200946665992, 2.6254064303178053, 0.24606060606060606)])
print(loss_)
# 可视化训练和验证的统计信息  
visualize_stats(loss_)