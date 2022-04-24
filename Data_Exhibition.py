import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# the meaning of data
# CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV

data = pd.read_csv(r"Data\housing.csv", names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                                               "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])

# draw picture
writer = SummaryWriter("X-Y axis data exhibition")
print(data)

# to tensor
data = np.array(data)
data = torch.Tensor(data)

array_x = [0.0 for i in range(len(data))]
array_y = [0.0 for i in range(len(data))]
array_x = np.array(array_x)
array_y = np.array(array_y)

# print(array_x)

names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

pos = 0
for name in names:
    for i in range(len(data)):
        array_x[i] = data[i][pos]
        array_y[i] = data[i][13]

    # print(array_x)
    plt.scatter(array_x, array_y)
    plt.xlabel(name)
    plt.ylabel("MEDV")
    plt.savefig("Exhibition_Picture/"+name+".png")
    plt.show()
    pos = pos +1


# for i in range(len(data)):
#     array_x[i] = data[i][0]
#     array_y[i] = data[i][13]
#
# print(array_x)
# plt.scatter(array_x, array_y)
# plt.title("per capita crime rate by town ")
# plt.savefig(fname=)
# plt.show()








#writer = SummaryWriter("Data_Picture")
