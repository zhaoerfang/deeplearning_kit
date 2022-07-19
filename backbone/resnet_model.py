import torch
import torch.nn as nn
import torchvision

# res18_torch = torchvision.models.resnet18()
# print(res18_torch)

# alex_torch = torchvision.models.AlexNet()
# print(alex_torch)

# def func(*args, **kwargs):
#     print(f"args type:\t {type(args)}\nargs:\t {args}")
#     print(f"args element:\t {args[0]}")
#     print("*" * 50)
#     print(f"kwargs type:\t {type(kwargs)}\nargs:\t {kwargs}")
#     print(f"kwargs element:\t {kwargs[list(kwargs.keys())[0]]}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)

