import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = [
    'resnet18_cifar',
    'resnet34_cifar',
    'resnet50_cifar',
    'resnet101_cifar',
    'resnet152_cifar',
]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :param groups: number of groups for group convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 1x1 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     padding=0, bias=False)

class DownConv(nn.Module):
    """
    Convolutional downsampling block
    """
    def __init__(self, channel_in, channel_out):
        super(DownConv, self).__init__()
        self.op = nn.Sequential(
            conv3x3(channel_in, channel_in, stride=2, groups=channel_in),
            conv1x1(channel_in, channel_in, stride=1),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            conv3x3(channel_in, channel_in, stride=1, groups=channel_in),
            conv1x1(channel_in, channel_out, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.op(x)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, nclasses=100, args=None):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.stage4 = self._make_layer(block, 256, layers[2], 2)
        self.stage5 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion

        self.proj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, 128)
        )

        self.multi_heads = nn.ModuleList([
        ])

    @property
    def n_params(self):
        return sum(np.prod(p.size()) for p in self.parameters())

    def _make_attention_layer(self, in_channels):
        layers = [
            DownConv(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def add_head(self, n_classes, bias=False):
        self.multi_heads.append(nn.Linear(self.out_dim, n_classes, bias=bias).cuda())

    def features(self, x):
        x0 = self.conv1(x)
        x = self.stage2(x0)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def head(self, x, use_proj=False):
        if use_proj:
            return self.proj_head(x)
        else:
            return torch.cat([head(x) for head in self.multi_heads], dim=1)
        
    def forward(self, x, use_proj=False):
        features = self.features(x)
        outputs = self.head(features, use_proj)

        if use_proj:
            return features, outputs
        else:
            return outputs
        

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        pass
    return model


def resnet18_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152_cifar(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)


if __name__ == "__main__":
    model = resnet18_cifar()
    print(model)