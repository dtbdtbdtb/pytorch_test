"""
# @Author: ros407
# @Date: 2024/12/26 17:56
# @Description: 
"""
import torch.nn as nn
import torch.nn.functional as F     #其中提供了许多神经网络中常用的函数，如激活函数、损失函数等，方便在模型中直接调用。

"""
定义LeNet模型，它继承了 PyTorch 中的 nn.Module 类，这使得它可以利用 PyTorch 提供的各种功能，如参数管理、自动微分等。
自动管理模型的可训练参数。
简化前向传播的定义。
方便地将模型移到 GPU 上进行加速。
使用 state_dict 轻松保存和加载模型。
组织复杂的模型结构。
与自动微分系统无缝集成，自动进行梯度计算和优化。
nn.Module 为 PyTorch 提供了一个强大且灵活的基础，简化了模型构建和训练的过程。
"""

class LeNet(nn.Module):
    def __init__(self):                 # 初始化函数，用于定义模型的结构。
        super(LeNet, self).__init__()   #初始化父类 nn.Module，以便可以使用 PyTorch 提供的功能。
        self.conv1 = nn.Conv2d(3, 16, 5) # 卷积层，输入通道数为3，输出通道数为16，卷积核大小为5*5，在模型定义时不需要指定bath_size，其在数据加载时指定。
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层，池化核大小为2*2，步长为2。
        self.conv2 = nn.Conv2d(16, 32, 5) # 卷积层，输入通道数为16，输出通道数为32，卷积核大小为5*5。
        self.pool2 = nn.MaxPool2d(2, 2) # 池化层，池化核大小为2*2，步长为2。
        self.fc1 = nn.Linear(32*5*5, 120) # 全连接层，输入维度为32*5*5，输出维度为120。
        self.fc2 = nn.Linear(120, 84) # 全连接层，输入维度为120，输出维度为84。
        self.fc3 = nn.Linear(84, 10) # 全连接层，输入维度为84，输出维度为10。

    def forward(self, x):               # 前向传播函数，定义模型的计算过程。输入是一个张量 x，包含batch_size,channel,height,width。
        x = F.relu(self.conv1(x))       # input: 3*32*32 -> output: 16*28*28 (32-5+2*0)/1+1
        x = self.pool1(x)               # input: 16*28*28 -> output: 16*14*14
        x = F.relu(self.conv2(x))       # input: 16*14*14 -> output: 32*10*10 (14-5+2*0)/1+1
        x = self.pool2(x)               # input: 32*10*10 -> output: 32*5*5
        x = x.view(-1, 32*5*5)          # 将张量展平为一维向量，方便全连接层处理。将形状为 [batch_size, 32, 5, 5] 的张量展平为 [batch_size, 800]，-1表示自动计算batch_size。
        x = F.relu(self.fc1(x))         # 全连接层，输入维度为800，输出维度为120。
        x = F.relu(self.fc2(x))         # 全连接层，输入维度为120，输出维度为84。
        x = self.fc3(x)                 # 全连接层，输入维度为84，输出维度为10。
        return x

# import torch #torch 是 PyTorch 的核心模块，提供了基本的张量操作、数学运算、优化算法等功能。它包含了如 Tensor 的创建、矩阵运算、线性代数、随机数生成等基础设施。torch 和 torch.nn 是两个不同的模块，它们分别包含了不同的功能。
# input1 = torch.rand([32, 3, 32, 32]) # 随机生成一个形状为 [32, 3, 32, 32] 的张量，用于测试模型。
# model = LeNet() # 创建一个 LeNet 模型。
# print(model)
# output = model(input1)