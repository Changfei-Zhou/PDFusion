import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_restormer import Restormer

EPSILON = 1e-10

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.leaky_relu(out, inplace=True)
        else:
            out = F.tanh(out)
        return out



# Convolution operation
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0_u = ConvLayer(1, 16, 1, 1, False)
        self.conv1_u = ConvLayer(16, 16, 3, 1, False)
        self.conv2_u = ConvLayer(32, 32, 3, 1, False)
        self.conv3_u = ConvLayer(64, 64, 3, 1, False)
        self.conv4_u = ConvLayer(128, 128, 3, 1, False)

        # self.conv1_f = ConvLayer(32, 16, 3, 1, False)
        # self.conv2_f = ConvLayer(64, 32, 3, 1, False)
        # self.conv3_f = ConvLayer(128, 64, 3, 1, False)
        # self.conv4_f = ConvLayer(256, 128, 3, 1, False)
        self.conv1_f = Restormer(32, 16, 16)
        self.conv2_f = Restormer(64, 32, 32)
        self.conv3_f = Restormer(128, 64, 64)
        self.conv4_f = Restormer(256, 128, 128)

        self.conv0_o = ConvLayer(1, 16, 1, 1, False)
        self.conv1_o = ConvLayer(16, 16, 3, 1, False)
        self.conv2_o = ConvLayer(32, 32, 3, 1, False)
        self.conv3_o = ConvLayer(64, 64, 3, 1, False)
        self.conv4_o = ConvLayer(128, 128, 3, 1, False)

        self.conv1_out = ConvLayer(128, 64, 3, 1, False)
        self.conv2_out = ConvLayer(64, 32, 3, 1, False)
        self.conv3_out = ConvLayer(32, 16, 3, 1, False)
        self.conv4_out = ConvLayer(16, 1, 3, 1, True)

    def forward(self, under, over):
        u0 = self.conv0_u(under)
        o0 = self.conv0_o(over)

        u1 = self.conv1_u(u0)
        o1 = self.conv1_o(o0)
        f1 = self.conv1_f(torch.cat([u1, o1], 1))


        u2 = self.conv2_u(torch.cat([u1, f1], 1))
        o2 = self.conv2_o(torch.cat([o1, f1], 1))
        f2 = self.conv2_f(torch.cat([u2, o2], 1))

        u3 = self.conv3_u(torch.cat([u2, f2], 1))
        o3 = self.conv3_o(torch.cat([o2, f2], 1))
        f3 = self.conv3_f(torch.cat([u3, o3], 1))

        u4 = self.conv4_u(torch.cat([u3, f3], 1))
        o4 = self.conv4_o(torch.cat([o3, f3], 1))
        f4 = self.conv4_f(torch.cat([u4, o4], 1))

        out = self.conv1_out(f4)
        out = self.conv2_out(out)
        out = self.conv3_out(out)
        out = self.conv4_out(out)

        return out


if __name__ == '__main__':
    tensor_variable = torch.randn(2, 1, 255, 256)
    net = Net()
    out = net(tensor_variable, tensor_variable)
    # 计算参数数量
    total_params = sum(p.numel() for p in net.parameters())
    print("Total Parameters:", total_params)
    print(out)
    print(out.shape)  # 打印网络输入的图像结构

