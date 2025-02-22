import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch
from random import sample
class ConvBNRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNRelu3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBNRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNRelu2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Enc_AAE(nn.Module):
    def __init__(self, channel=15, output_dim=4096, windowSize=25):
        super(Enc_AAE, self).__init__()
        self.channel = channel
        self.output_dim = output_dim
        self.windowSize = windowSize
        ##(N,1,15,25,25)
        # 3D卷积降低波段数
        self.conv1 = ConvBNRelu3D(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = ConvBNRelu3D(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        #(N,16,8,13,13)   
        self.conv3 = ConvBNRelu3D(in_channels=16, out_channels=32, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        #(N,32,4,7,7)  
        # 将3D卷积的输出reshape为2D卷积输入
        self.conv4 = ConvBNRelu2D(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv5 = ConvBNRelu2D(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        # 添加池化层
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层将特征向量转换为输出维度
        self.projector = nn.Sequential(
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        x = self.conv1(x)  # (N, 16, 8, windowSize, windowSize)
        x = self.conv2(x)  # (N, 32, 4, windowSize, windowSize)
        x = self.conv3(x)  # (N, 64, 2, windowSize, windowSize)

        # 调整形状为2D卷积输入
        x = x.view(x.size(0), -1, x.size(3), x.size(4))  # (N, 64 * 2, windowSize, windowSize)
        
        x = self.conv4(x)  # (N, 128, windowSize/2, windowSize/2)
        x = self.conv5(x)  # (N, 128, windowSize/4, windowSize/4)

        # 池化层
        x = self.pool(x)  # (N, 128, 4, 4)

        # 全连接层
        x = x.view(x.size(0), -1)  # 展平
        h = x
        mu = self.projector(x)
        return h, mu


class DeconvBNRelu2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(DeconvBNRelu2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class DeconvBNRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(DeconvBNRelu3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class Dec_AAE(nn.Module):
    def __init__(self, input_dim=2048, channel=15, windowSize=25):
        super(Dec_AAE, self).__init__()
        self.channel = channel
        self.input_dim = input_dim
        self.windowSize = windowSize

        # 全连接层将输入维度转换为特征向量
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # 反池化层
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#(N, 256 * 8 * 8)

        # 2D反卷积层
        self.deconv5 = DeconvBNRelu2D(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=0, output_padding=0)
        #(N, 128 * 19 * 19)
        self.deconv4 = DeconvBNRelu2D(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=0, output_padding=0)
        #(N, 128 * 25 * 25)
        # 将2D反卷积的输出reshape为3D反卷积输入
        #(N, 64，2，25，25)
        self.deconv3 = DeconvBNRelu3D(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), output_padding=(1, 0, 0))
        #(N, 32，4，25，25)

        self.deconv2 = DeconvBNRelu3D(in_channels=32, out_channels=16, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), output_padding=(1, 0, 0))
        #(N, 16，8，25，25)

        # 最后一层反卷积层，不使用激活函数
        self.deconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1), output_padding=(0, 0, 0))
        #(N, 1，15，25，25)
        #添加BN层
        # self.bn = nn.BatchNorm3d(1)
    def forward(self, x):
        # 全连接层
        x = self.projector(x)  # (N, 256 * 4 * 4)
        x = x.view(x.size(0), 256, 4, 4)  # (N, 256, 4, 4)

        # 反池化层
        x = self.unpool(x)  # (N, 256, 16, 16)

        # 2D反卷积层
        x = self.deconv5(x)  # (N, 128, 32, 32)
        x = self.deconv4(x)  # (N, 64 * (channel / 8), 64, 64)

        # 调整形状为3D反卷积输入
        x = x.view(x.size(0), 64, 2, self.windowSize, self.windowSize)  # (N, 64, channel / 8, windowSize, windowSize)

        # 3D反卷积层
        x = self.deconv3(x)  # (N, 32, channel / 4, windowSize, windowSize)
        x = self.deconv2(x)  # (N, 16, channel / 2, windowSize, windowSize)
        x = self.deconv1(x)  # (N, 1, channel, windowSize, windowSize)
        # x = self.bn(x)
        return x


class Discriminant(nn.Module):
    def __init__(self,encoded_dim):
        super(Discriminant, self).__init__()
        self.lin1 = nn.Linear(encoded_dim, 1024)
        self.relu=nn.ReLU(inplace=False)
        self.lin2 = nn.Linear(1024, 128)
        self.relu2=nn.ReLU(inplace=False)
        self.lin3 = nn.Linear(128,1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = self.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = self.relu2(self.lin3(x))
        return torch.tanh(x)


class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.fc1 = nn.Linear(n_features,128 )
        self.relu1=nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2=nn.ReLU()
        self.fc3 = nn.Linear(64, n_classes)
    def forward(self, x):
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)

        return x

