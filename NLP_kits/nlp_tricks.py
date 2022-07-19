from argon2 import Parameters
import torch
import torch.nn as nn
import torch.nn.functional as F


class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 这里卷积核的大小是 5, 个数是 6, 输入的 width 是 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 两次卷积的结果应该是 5x5x16 的矩阵
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # 可以看出网络层的结构, 两个卷积层, 其余还有全连接层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_param():
    my_net = TheModelClass()
    print("this is model.parameters():")
    print("model.parameters.type:\t", type(my_net.parameters()))
    print("params.type:\t", type(next(my_net.parameters())))
    for param in my_net.parameters():
        print("param.type2:\t", type(param))
        break


def grad_clipping(net: nn.Module, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


if __name__ == "__main__":
    ...
