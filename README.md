# Hand gesture recognition with FMCW
Model pre-trained on [Deep soli dataset](https://github.com/simonwsw/deep-soli/tree/master) and fine-tuned on self-collected dataset recorded using BGT60TR13C 60 GHz FMCW radar. It consists of only 350 classes over 7 classes recorded by only one person (the dataself is not included). Therefore the model is not production-ready. The model itself is a combination of CNN for spacial features extraction and LSTM for temporal features extraction.


# NOTE
This project is copied from `https://github.com/4uf04eG/FMCW-gesture-recognitio`

___

#  基于FMCW雷达的手势识别模型

## 数据集[deep-soli](https://github.com/simonwsw/deep-soli/tree/master)

![](images/2025-02-22-19-14-18.png)

数据经过预处理后，格式为[frameNum, channalNum, 32, 32]，其中FrameNum是一次动作采集的帧数，channalNum是接收天线通道数，此处一共有4个接收天线，不过训练数据之只有了前三个通道，32×32就是RangeDopplerMap的大小。


## 数据范例
Slow swipe

![DEMO-slow swipe](images/4_4_46.gif)

Push

![DEMO-slow swipe](images/9_13_9.gif)

## 模型结构


模型分为两个部分：CNN+LSTM，结构如下：

![](images/2025-02-22-19-12-59.png)
