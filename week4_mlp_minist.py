#-*- coding: utf-8 -*-
# 1.课堂代码复现：使用MLP对 手写数字MNIST进行识别
# 2.学会使用pytorch从加载数据，模型搭建、模型训练、预测，整套流程。

import torch
from torch import nn
import torchvision
from torchvision import datasets,transforms

#****************数据加载******************

#一个批次加载的图片数量
batch_size = 32
#数据预处理
#Compose用于将多个transform组合起来
#ToTensor()将像素转换为tensor,并做Min-max归一化，即x1 = x-min/max_min,相当于将像素从[0,255]转换为[0,1]
# Normalize()用均值和标准差对图像标准化处理 x'=(x-mean)/std，加速收敛的作用
# 这里0.131是图片的均值，0.308是方差，通过对原始图片进行计算得出
# 想偷懒的话可以直接填Normalize([0.5], [0.5])
# 另外多说一点，因为MNIST数据集图片是灰度图，只有一个通道，因此这里的均值和方差都只有一个值
# 若是普通的彩色图像，则应该是三个值，比如Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.131],[0.308])])

# 下载数据集
# 训练数据集 train=True
# './data/mnist'是数据集存放的路径，可自行调整
# download=True表示叫pytorch帮我们自动下载
data_train = datasets.MNIST('./data/mnist',
                            train=True,
                            transform=mnist_transforms,
                            download=True
                            )
# 测试数据集 train=False
data_test = datasets.MNIST('./data/mnist',
                           train=False,
                           transform=mnist_transforms,
                           download=True
                           )

#加载数据集
# shuffle=True表示加载时打乱图片顺序，有一定的防止过拟合效果
loader_train = torch.utils.data.DataLoader(data_train,
                                           batch_size = batch_size,
                                           shuffle = True)

# 测试集就不需要打乱了，因此shuffle=False
loader_test = torch.utils.data.DataLoader(data_test,
                                          batch_size=batch_size,
                                          shuffle=False)

#**************************模型定义******************************
# 定义模型
# MLP (
# (fc1): Linear (784 -> 512)
# (fc2): Linear (512 -> 128)
# (fc3): Linear (128 -> 10)
# )
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,input):
        input = input.view(-1,28*28)
        input = nn.functional.relu(self.fc1(input))
        input = nn.functional.relu(self.fc2(input))
        return nn.functional.softmax(self.fc3(input))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)
print(model)

#**************************训练***********************************

def train(model):
    model.train()
    #设置学习率
    #learning_rate = 1e-6
    #设置epoch
    epoch = 5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    for i in range(epoch):
        #网络训练
        # 1.初始化，清空网络内上一次训练得到的梯度
        # 2.载入数据，送入网络进行前向传播
        # 3.计算代价函数，并进行反向传播计算梯度
        # 4.调用优化器进行优化
        train_loss = 0.0
        for data, target in loader_train:
            optimizer.zero_grad()#清空上一步的残余更新参数值
            output= model(data)
            loss = loss_func(output,target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*data.size(0)
        train_loss = train_loss / len(loader_train)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(i + 1, train_loss))
# 在数据集上测试神经网络
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 训练集中不需要反向传播
        for data in loader_test:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100.0 * correct / total
