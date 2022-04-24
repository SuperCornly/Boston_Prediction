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

if __name__ == '__main__':

    # load model
    BP = torch.load("SavedModel/BP.pkl")
    print(BP.parameters)

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

    # 用测试集数据，验证测试效果
    y_test_pred = BP.forward(x_test)

    loss_test = torch.mean((y_test_pred - y_test) ** 2)
    print("loss for test:", loss_test.data)

    plt.scatter(range(len(y_test)), y_test.data, color="blue")
    plt.scatter(range(len(y_test_pred)), y_test_pred.data, color="red")
    plt.title("hourse value Comparison")
    plt.show()