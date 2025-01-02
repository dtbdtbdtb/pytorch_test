"""
# @Author: ros407
# @Date: 2024/12/26 21:37
# @Description: 
"""
import torch
import torchvision.transforms as transforms
from model import LeNet
from PIL import Image

# 选择设备：检查是否有可用的 GPU，若没有，则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),                #将图像resize为32*32以适应网络输入
    transforms.ToTensor(),                      #将输入的图像转换为PyTorch张量（Tensor）
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #对张量进行归一化处理，使用均值 (0.5, 0.5, 0.5) 和标准差 (0.5, 0.5, 0.5) 对每个通道进行标准化
])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #元组类型，不可改变

# 加载模型并迁移到设备
net = LeNet().to(device)  # 将模型迁移到 GPU 或 CPU
net.load_state_dict(torch.load('Lenet.pth'))  # 加载模型权重

im = Image.open('1.jpg')
im = transform(im) #将图像转换为PyTorch张量[C H W]
im = torch.unsqueeze(im, dim=0) #将张量转换为[1 C H W] 1表示batch_size，就是给它整一个batch维度

# 将输入图像迁移到设备（GPU 或 CPU）
im = im.to(device)

with torch.no_grad():
    outputs = net(im)
    predict_score = torch.softmax(outputs, dim=1)
    predict = torch.max(outputs.data, dim=1)[1].cpu().numpy() #将输出转换为预测值

print(predict_score)
print(classes[int(predict)])
