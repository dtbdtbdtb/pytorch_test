"""
# @Author: ros407
# @Date: 2024/12/26 19:09
# @Description: 
"""
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim                     #torch.optim 是 PyTorch 中用于优化神经网络的模块。它提供了各种常用的优化算法，比如 SGD（随机梯度下降）、Adam、RMSprop 等。优化器通常用于 更新模型参数，通过最小化损失函数来训练神经网络模型。
import torchvision.transforms as transforms     #torchvision.transforms 提供了多种常见的图像预处理操作，通常在加载图像数据时使用。图像预处理有助于标准化图像尺寸、转换颜色通道、数据增强等。
import matplotlib.pyplot as plt                 #matplotlib.pyplot 是 Matplotlib 库的一个模块，它提供了一系列函数来创建各种图形、可视化数据。plt 是 pyplot 模块的常用别名，通常用于绘制 2D 图形（如折线图、散点图、直方图、饼图等）。
import numpy as np                              #numpy 是 Python 中进行科学计算的基础库，提供了 高效的多维数组操作（ndarray）以及对数值计算（如线性代数、傅里叶变换、随机数生成等）的支持。

"""
对于使用GPU进行加速的步骤
1.检查是否有可用的 GPU：使用 torch.cuda.is_available() 来检查是否有 GPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
2.将模型迁移到 GPU：通过 .to(device) 方法，将模型移到 GPU 上。
    net = LeNet().to(device)
3.将输入数据和标签迁移到 GPU：每次获取数据时，需要将输入和标签也移到 GPU 上。
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
"""

# 选择设备：检查是否有可用的 GPU，若没有，则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# 定义了一个图像预处理转换管道,将多个图像处理操作串联在一起。
1.在 PyTorch 中，神经网络需要的输入数据是 Tensor（而不是图像文件），
并且图像的像素值通常需要被归一化处理。这个转换会把图像从 PIL 图像 或 NumPy 数组 (H x W x C)转换成 Tensor 格式(C x H x W)。
另外，它会自动将图像的像素值 缩放到 [0, 1] 范围，因为原图像的像素值通常是 [0, 255]，
而 Tensor 的默认值是浮动在 [0, 1] 之间。
2.对于每个通道（RGB），均值为 0.5，标准差也为 0.5，这相当于把每个通道的像素值缩放到 [-1, 1] 范围内。
"""

transform = transforms.Compose([
    transforms.ToTensor(),                      #将输入的图像转换为PyTorch张量（Tensor）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #对张量进行归一化处理，使用均值 (0.5, 0.5, 0.5) 和标准差 (0.5, 0.5, 0.5) 对每个通道进行标准化
])

# 50000张训练图片
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,      #导入数据集存放的地址，true表示导入训练集
                                        download=False, transform=transform) #true表示自动下载，再通过transform进行预处理
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,      #将数据集转换成一批一批的，批量大小为36
                                          shuffle=True, num_workers=2)  #随机打乱数据集，使用2个进程进行数据加载
# 10000张测试图片
testset = torchvision.datasets.CIFAR10(root='./data', train=False,      #导入数据集存放的地址，false表示导入验证集
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=2)
test_data_iter = iter(testloader) #创建一个迭代器，用于获取测试数据
test_image, test_label = test_data_iter.next() #从迭代器中获取下一组测试数据
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #元组类型，不可改变
#查看数据集中的图片
# def imshow(img):
#     img = img / 2 + 0.5     # 反标准化:input*std+mean（0，1），标准化:(input - mean)/std（-1，1）;
#     npimg = img.numpy()     # 将张量转换为 numpy 数组
#     plt.imshow(np.transpose(npimg, (1, 2, 0))) # 将 numpy 数组(C x H x W)转换为 plt 可识别的格式(H x W x C)
#     plt.show() # 显示图片imshow()：将图像数据绘制到当前的图形区域。show()：打开一个交互式窗口，显示所有绘制的内容，包括 imshow() 渲染的图像。
#
# # print labels  打印一组图像的标签
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# # show images   显示一组图像
# imshow(torchvision.utils.make_grid(test_image)) # make_grid() 函数将一个 batch 的图像拼成一张大图，方便显示

# net = LeNet() # 实例化一个 LeNet 模型
# 确保将模型迁移到 GPU（如果可用）
net = LeNet().to(device)  # 将 LeNet 模型移动到 GPU 或 CPU

loss_function = nn.CrossEntropyLoss() # 定义损失函数为交叉熵损失函数
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 定义优化器为 SGD 优化器(模型的所有参数，学习率，动量)
optimizer = optim.Adam(net.parameters(), lr=0.001) # 定义优化器为 Adam 优化器(模型的所有参数，学习率)

# 训练模型
for epoch in range(20):

    running_loss = 0.0 # 训练损失初始化为0
    #这个变量用来存储当前训练周期（epoch）中的累积损失值。每次训练步（iteration）完成后，你可以将当前的损失加到 running_loss 中，然后在每个 epoch 完成时计算平均损失值。
    for step, data in enumerate(trainloader, start=0): # 遍历训练数据集 enumerate() 是 Python 内置的一个函数，用于遍历可迭代对象（如列表、元组、迭代器等），它会为每个元素返回两个值：元素的索引（step）和元素本身（data）。
        # 获取输入和标签
        inputs, labels = data # data 是一个元组，包含输入和标签
        inputs, labels = inputs.to(device), labels.to(device)# 将数据和标签迁移到设备（GPU 或 CPU）

        # zero the parameter gradients
        optimizer.zero_grad() # 梯度清零
        #在每一次反向传播前，必须将模型参数的梯度清零。PyTorch 默认会在每次 .backward() 后累积梯度，因此你需要手动清除它们，以避免梯度累加到前一个批次的梯度上。
        # forward + backward + optimize
        outputs = net(inputs) # 前向传播，输出【batch，num_classes=10】
        loss = loss_function(outputs, labels) # 计算损失
        loss.backward() # 反向传播
        #这行代码更新模型参数。优化器（如 SGD, Adam 等）使用梯度信息来调整模型的权重，以最小化损失函数。step() 方法会根据每个参数的梯度值来更新模型权重。
        optimizer.step()

        # 打印训练信息
        running_loss += loss.item() # 将当前损失值加到 running_loss 上
        if step % 500 == 499:       # 每 500 个批次打印一次损失值
            with torch.no_grad():   # 以下步骤不进行梯度计算，如果不用那么在验证阶段也会计算损失梯度：占用算力，存储梯度占用内存会导致验证测试过程内存爆满
                # 验证模型
                # 将测试数据迁移到 GPU 或 CPU
                test_image, test_label = test_image.to(device), test_label.to(device)
                outputs = net(test_image) # 前向传播 输出【batch，10】
                predict_y = torch.max(outputs.data, dim=1)[1] #输出的最大值索引（在第1维找，0维是batch）使用 .data 取出张量中的原始值，表示不需要对其进行梯度跟踪。
                #torch.max(input, dim) 是 PyTorch 的内置函数，用于沿指定维度找到最大值。它返回两个值：最大值的具体数值。最大值所在的索引（index）。[1]表示只返回最大值的索引，这些索引对应预测结果
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                """
                predict_y：这是模型对测试数据的预测结果。它是一个包含每张图片预测类别的张量，形状为 [batch_size]，其中每个元素表示模型对对应测试图片的预测类别索引（如 [2, 4, 1, 3, ...]）。
                test_label：这是测试集中的真实标签。它也是一个张量，形状为 [batch_size]，包含每张测试图片的真实类别索引（如 [2, 4, 1, 3, ...]）。
                predict_y == test_label：这部分代码会返回一个布尔值张量，表示模型预测的类别是否与真实标签一致。结果是一个形状为 [batch_size] 的布尔张量，True 为 1，False 为 0。
                
                对布尔张量执行 .sum() 操作，会把 True 转换为 1，False 转换为 0，然后计算所有元素的和，即预测正确的样本数量。
                .sum() 返回的是一个张量，.item() 将其转换为普通的 Python 数字。
                test_label.size(0)：test_label 是一个包含测试标签的张量，size(0) 返回的是测试集的样本数量，即批次的大小（batch_size）。
                """
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')
save_path = 'weights/Lenet.pth'  # 保存模型
torch.save(net.state_dict(), save_path)