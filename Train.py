import Module
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt  # 画图库
import torch  # torch基础库
import torch.nn as nn  # torch神经网络库
import torch.nn.functional as F  # torch神经网络库
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# print(BP)

if __name__ == '__main__':

    # 2-1 准备数据集
    full_data = load_boston()

    # 提前样本数据
    x_raw_data = full_data['data']
    print(x_raw_data.shape)

    # 提取样本标签
    y_raw_data = full_data['target']
    print(y_raw_data.shape)
    y_raw_data = y_raw_data.reshape(-1, 1)
    print(y_raw_data.shape)

    # 房价的波动范围
    plt.scatter(range(len(y_raw_data)), y_raw_data, color="blue")
    plt.title("hourse value")
    plt.show()

    # 2-2 对数据预处理
    print("数据规范化预处理，形状不变，内容改变")
    ss = MinMaxScaler()
    x_data = ss.fit_transform(x_raw_data)
    y_data = y_raw_data
    print(x_data.shape)
    print(y_data.shape)

    print("\n把x、y从ndarray格式转成torch格式")
    x_sample = torch.from_numpy(x_data).type(torch.FloatTensor)
    y_sample = torch.from_numpy(y_data).type(torch.FloatTensor)
    print(x_sample.shape)
    print(y_sample.shape)

    print("\n数据集切割成：训练数据集和测试数据集")
    # 0.2 表示测试集占比
    x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Module
    BP = Module.BPModel()
    print(BP)
    print(BP.parameters)

    # 2-4 定义网络预测输出
    y_pred = BP.forward(x_train)
    print(y_pred.shape)

    # 3-1 定义loss函数:
    # loss_fn= MSE loss
    loss_fn = nn.MSELoss()

    print(loss_fn)

    # 3-2 定义优化器
    Learning_rate = 0.01  # 学习率

    # optimizer = SGD： 基本梯度下降法
    # parameters：指明要优化的参数列表
    # lr：指明学习率
    optimizer = torch.optim.SGD(BP.parameters(), lr=Learning_rate)
    print(optimizer)

    # 3-3 模型训练
    # 定义迭代次数
    epochs = 100000

    loss_history = []  # 训练过程中的loss数据
    y_pred_history = []  # 中间的预测结果

    for i in range(0, epochs):

        # (1) 前向计算
        y_pred = BP(x_train)
        # (2) 计算loss
        loss = loss_fn(y_pred, y_train)
        # (3) 反向求导
        loss.backward()
        # (4) 反向迭代
        optimizer.step()
        # (5) 复位优化器的梯度
        optimizer.zero_grad()
        # 记录训练数据
        loss_history.append(loss.item())
        y_pred_history.append(y_pred.data)

        if (i % 10000 == 0):
            print('epoch {}  loss {:.4f}'.format(i, loss.item()))

    print("\n迭代完成")
    print("final loss =", loss.item())
    print(len(loss_history))
    print(len(y_pred_history))

    # 显示loss的历史数据
    plt.plot(loss_history, "r+")
    plt.title("loss value")
    plt.show()

    # 显示loss的历史数据
    plt.plot(loss_history[1000::], "r+")
    plt.title("loss value")
    plt.show()


    # 用训练集数据，检查训练效果
    y_train_pred = BP.forward(x_train)

    loss_train = torch.mean((y_train_pred - y_train) ** 2)
    print("loss for train:", loss_train.data)

    plt.scatter(range(len(y_train)), y_train.data, color="blue")
    plt.scatter(range(len(y_train_pred)), y_train_pred.data, color="red")
    plt.title("hourse value")
    plt.show()

    # 用测试集数据，验证测试效果
    y_test_pred = BP.forward(x_test)

    loss_test = torch.mean((y_test_pred - y_test) ** 2)
    print("loss for test:", loss_test.data)

    plt.scatter(range(len(y_test)), y_test.data, color="blue")
    plt.scatter(range(len(y_test_pred)), y_test_pred.data, color="red")
    plt.title("hourse value")
    plt.show()

    # 存储模型
    # torch.save(BP, "SavedModel/BP.pkl")


