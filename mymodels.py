import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms, datasets
import warnings


def LeNetmodel():
    load_model_start = time.time()

    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        # nn.Sigmoid(),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=4, stride=4),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10000),
        nn.ReLU(),
        nn.Linear(10000, 500),
        nn.ReLU(),
        nn.Linear(500, 10))

    print(model)

    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model

def VGGmodel():
    load_model_start = time.time()

    def vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    def vgg(conv_arch):
        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 8 * 8, 4096), nn.ReLU(), nn.Dropout(0.5),
            # nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 10))

    model = vgg(conv_arch)


    print(model)

    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model

def NiNmodel():
    load_model_start = time.time()

    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

    model = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),  # 全局的平均池化层
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())
    print(model)

    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model

def GoogLeNetmodel():
    load_model_start = time.time()

    class Inception(nn.Module):
        # c1--c4是每条路径的输出通道数
        def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # 线路1，单1x1卷积层
            self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
            # 线路2，1x1卷积层后接3x3卷积层
            self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
            self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            # 线路3，1x1卷积层后接5x5卷积层
            self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
            self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            # 线路4，3x3最大汇聚层后接1x1卷积层
            self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

        def forward(self, x):
            p1 = F.relu(self.p1_1(x))
            p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            p4 = F.relu(self.p4_2(self.p4_1(x)))
            # 在通道维度上连结输出
            return torch.cat((p1, p2, p3, p4), dim=1)

    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())

    model = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    print(model)
    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model

def ResNetmodel():
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

def CBAM_ResNetmodel():
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
                       nn.BatchNorm2d(64), nn.ReLU(),b7,
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
    # b6 = net.forward()

    model = nn.Sequential(b1, b2, b3, b4, b5,b6,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    print(model)
    X = torch.randn(7, 1, 256, 256)
    for layer in model:
        X = layer(X)
    load_model_end = time.time()
    print("===============================> Time for loading model: {:.4f}s".format(load_model_end - load_model_start))

    return model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 16 * 256, 2560)
        # self.fc2 = nn.Linear(2560, 4)
        self.fc2 = nn.Linear(2560, 10)

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.size())
        x = x.view(-1, 16 * 16 * 256)
        # print(x.size())
        # x = self.Dropout(0.6)(x)
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x


class d1cnn(nn.Module):
    def __init__(self):
        super(d1cnn, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,100), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 100), padding=1)
        # self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 100), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 100), padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 1 * 37, 2560)
        # self.fc2 = nn.Linear(2560, 4)
        self.fc2 = nn.Linear(2560, 10)

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.size())
        x = x.view(-1, 256 * 1 * 37)
        # print(x.size())
        # x = self.Dropout(0.6)(x)
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.fc2(x)
        # print(x.size())
        return x


def CNN2():
    load_model_start = time.time()

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(8, 16, kernel_size=3,padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 65 * 65, 32),
        nn.ReLU(),
        # nn.Linear(10000, 500),
        # nn.ReLU(),
        nn.Linear(32, 10))

    print(model)
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

class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 64
        self.kernel_num = 16
        self.num_layers = 2
        self.V = 5
        self.embed1 = nn.Sequential(
            nn.Conv2d(in_channel, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.embed2 = nn.Sequential(
            nn.Conv2d(self.kernel_num, self.kernel_num*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num*2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(self.V))
        self.hidden2label1 = nn.Sequential(nn.Linear(self.V*self.V * 2 * self.hidden_dim, self.hidden_dim * 4), nn.ReLU(), nn.Dropout())
        self.hidden2label2 = nn.Linear(self.hidden_dim * 4, out_channel)
        self.bilstm = nn.LSTM(self.kernel_num*2, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, bias=False)

    def forward(self, x):
        x = self.embed1(x)
        x = self.embed2(x)
        x = x.view(-1, self.kernel_num*2, self.V*self.V)
        x = torch.transpose(x, 1, 2)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        # bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)
        bilstm_out = torch.reshape(bilstm_out,(bilstm_out.size(0), -1))
        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)
        # model = self.hidden2label2(logit)


        return logit

class AlexNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_channel),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class CNN3(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN3, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            # nn.Conv2d(in_channel, 16, kernel_size=3),  # 16, 26 ,26
            nn.Conv2d(in_channel, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.Tanh())
            # nn.Sigmoid())
            # nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            # nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1
            nn.MaxPool2d(kernel_size=4, stride=4))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4,4)))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.Tanh())
            # nn.Sigmoid())
            # nn.ReLU(inplace=True))
        self.fc = nn.Linear(128, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x

class D1CNN3(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(D1CNN3, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=(1,8)),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            # nn.Tanh())
            # nn.Sigmoid())
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1,16)),  # 32, 24, 24
            nn.BatchNorm2d(32),
            # nn.Tanh())
            # nn.Sigmoid())
            nn.ReLU(inplace=True))
            # nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,64)),  # 64,10,10
            nn.BatchNorm2d(64),
            # nn.Tanh())
            # nn.Sigmoid())
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,128)),  # 128,8,8
            nn.BatchNorm2d(128),
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4,4)))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            # nn.Tanh())
            # nn.Sigmoid())
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(128, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x

class LeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((5, 5))  # adaptive change the outputsize to (16,5,5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


