import torch
import torch.nn as nn
import numpy as np
import config as c


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bottleneck=False, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(ResBlock, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        if bottleneck:
            width = int(width / 2)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, numOfResBlock, input_shape, data_type='hoa'):
        super(ResNet, self).__init__()
        (inChannels, height, width) = input_shape
        self.data_type = data_type

        if self.data_type == 'hoa':
            self.conv1 = nn.Conv2d(inChannels, 64, kernel_size=(1, 7), stride=(1, 3))
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(inChannels, 128, kernel_size=(1, 7), stride=(1, 3))
            self.bn1 = nn.BatchNorm2d(128)

        if self.data_type == 'hoa':
            self.conv2 = nn.Conv2d(64, 128,  kernel_size=(1, 5), stride=(1, 2))
            self.bn2 = nn.BatchNorm2d(128)
        else:
            self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 5), stride=(1, 2))
            self.bn2 = nn.BatchNorm2d(256)

        self.ResBlock = block
        self.blockNum = numOfResBlock

        if self.data_type == 'hoa':
            self.conv3 = conv1x1(128, 360)
            self.bn3 = nn.BatchNorm2d(360)
        else:
            self.conv3 = conv1x1(256, 360)
            self.bn3 = nn.BatchNorm2d(360)

        self.conv4 = conv1x1(40, 128)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 1, kernel_size=(height, 3), padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _print = False
        if _print: print('input:', x.shape)
        out = self.conv1(x)
        if _print: print('after conv1', out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if _print: print('after conv2:', out.shape)
        for i in range(self.blockNum):
            out = self.ResBlock(out)
            if _print: print('after res block %d:' % i, out.shape)

        out = self.conv3(out)
        if _print: print('after conv3:', out.shape)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.swapaxis(out)  # change the freq axis and channel axis
        if _print: print('after swapping axes:', out.shape)
        out = self.conv4(out)
        if _print: print('after conv4:', out.shape)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        if _print: print('after conv5:', out.shape)

        out = self.relu(out)
        out = torch.squeeze(out)
        if _print: print('out shape:', out.shape)
        return out

    def swapaxis(self, x):
        return x.permute([0, 3, 2, 1])


class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.layer1 = nn.Sequential(
                            nn.Conv1d(5, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*80, c.frame_size)

    def forward(self, x):
        out = self.layer1(x)
        # print('经卷积层之后：', out.shape)
        out = out.reshape(out.size(0), -1)
        # print('reshape之后：', out.shape)
        out = self.fc(out)
        # print('经全连接之后：', out.shape)
        return out

class SENet1(nn.Module):
    '''
    此网络接受五个波束信号（1直达+4反射）的STFT谱，实部和虚部拼在一起，共十个输入通道
    '''
    def __init__(self):
        super(SENet1, self).__init__()
        self.layer1 = nn.Sequential(
                            nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*80, c.frame_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.layer1(x)
        # print('经第一层卷积层之后：', out.shape)
        out = self.layer2(x)
        # print('经第二层卷积层之后：', out.shape)
        out = out.reshape(out.size(0), -1)
        # print('reshape之后：', out.shape)
        out = self.fc(out)
        # print('经全连接之后：', out.shape)
        out = self.dropout(out)
        return out

class SENet2(nn.Module):
    '''
    此网络接受一个波束信号（1直达）的STFT谱，实部和虚部拼在一起，共2个输入通道
    '''
    def __init__(self):
        super(SENet2, self).__init__()
        self.layer1 = nn.Sequential(
                            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(8),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*80, c.frame_size)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        out = self.layer1(x)
        # print('经第一层卷积层之后：', out.shape)
        out = self.layer2(x)
        # print('经第二层卷积层之后：', out.shape)
        out = out.reshape(out.size(0), -1)
        # print('reshape之后：', out.shape)
        out = self.fc(out)
        # print('经全连接之后：', out.shape)
        out = self.dropout(out)
        return out


class BeamEnhancer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_v, device):
        super(BeamEnhancer, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_v, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def test_BeamEnhancer():
   test_data = torch.randn(32, 100, 129)
   model = BeamEnhancer(input_size=129, hidden_size=200, output_size=129, num_layers=2, device=torch.device('cpu'))
   output_data = model(test_data)
   print(output_data.shape) # (32, 100, 129)


if __name__ == '__main__':
    test_BeamEnhancer()
