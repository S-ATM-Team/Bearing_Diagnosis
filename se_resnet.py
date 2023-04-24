from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
import time
from torch.nn import functional as F
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_ResNetmodel():
    load_model_start = time.time()

    class Residual(nn.Module):  # @save
        def __init__(self, input_channels, num_channels,  # num_channels输出的通道数
                     use_1x1conv=False, strides=1,reduction=16):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels,
                                       kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            self.se = SELayer(num_channels, reduction)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            Y = self.se(Y)
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)


    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 高宽减半，通道数加倍
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # *resnet_block中*是解包
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))


    # b6 = net.forward()

    model = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    print(model)
    # X = torch.randn(7, 1, 256, 256)
    # for layer in model:
    #     X = layer(X)
    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

def cbam_ResNetmodel():
    load_model_start = time.time()

    class Residual(nn.Module):  # @save
        def __init__(self, input_channels, num_channels,  # num_channels输出的通道数
                     use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels,
                                       kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)

    b7 = CBAMLayer(64)
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(), b7,
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 高宽减半，通道数加倍
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # *resnet_block中*是解包
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    b6 = CBAMLayer(512)


    model = nn.Sequential(b1, b2, b3, b4, b5,b6,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    print(model)

    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model

def cbam2_ResNetmodel():
    load_model_start = time.time()

    class Residual(nn.Module):  # @save
        def __init__(self, input_channels, num_channels,  # num_channels输出的通道数
                     use_1x1conv=False, strides=1,reduction=16):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels,
                                       kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            # self.se = SELayer(num_channels, reduction)
            self.cbam = CBAMLayer(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            # Y = self.se(Y)
            Y = self.cbam(Y)
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)


    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 高宽减半，通道数加倍
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # *resnet_block中*是解包
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))


    # b6 = net.forward()

    model = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    print(model)
    # X = torch.randn(7, 1, 256, 256)
    # for layer in model:
    #     X = layer(X)
    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model


def se2_ResNetmodel():
    load_model_start = time.time()

    class Residual(nn.Module):  # @save
        def __init__(self, input_channels, num_channels,  # num_channels输出的通道数
                     use_1x1conv=False, strides=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
            self.conv2 = nn.Conv2d(num_channels, num_channels,
                                   kernel_size=3, padding=1)
            if use_1x1conv:
                self.conv3 = nn.Conv2d(input_channels, num_channels,
                                       kernel_size=1, stride=strides)
            else:
                self.conv3 = None
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)

        def forward(self, X):
            Y = F.relu(self.bn1(self.conv1(X)))
            Y = self.bn2(self.conv2(Y))
            if self.conv3:
                X = self.conv3(X)
            Y += X
            return F.relu(Y)

    b7 = SELayer(64, 16)
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(), b7,
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 高宽减半，通道数加倍
    b3 = nn.Sequential(*resnet_block(64, 128, 2))  # *resnet_block中*是解包
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    b6 = SELayer(512, 16)


    model = nn.Sequential(b1, b2, b3, b4, b5,b6,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    print(model)
    # X = torch.randn(7, 1, 256, 256)
    # for layer in model:
    #     X = layer(X)
    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model