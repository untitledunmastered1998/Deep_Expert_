# from typing import List
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = [
#     'resnet18_expert',
#     'resnet34_expert',
#     'resnet50_expert',
#     'resnet101_expert',
#     'resnet152_expert',
# ]


# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> F.conv2d:
#     """
#     Instantiates a 3x3 convolutional layer with no bias.
#     :param in_planes: number of input channels
#     :param out_planes: number of output channels
#     :param stride: stride of the convolution
#     :param groups: number of groups for group convolution
#     :return: convolutional layer
#     """
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, groups=groups, bias=False)


# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
#     """
#     Instantiates a 1x1 convolutional layer with no bias.
#     :param in_planes: number of input channels
#     :param out_planes: number of output channels
#     :param stride: stride of the convolution
#     :return: convolutional layer
#     """
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                      padding=0, bias=False)


# class DownConv(nn.Module):
#     """
#     Convolutional downsampling block
#     """

#     def __init__(self, channel_in, channel_out):
#         super(DownConv, self).__init__()
#         self.op = nn.Sequential(
#             conv3x3(channel_in, channel_in, stride=2, groups=channel_in),
#             conv1x1(channel_in, channel_in, stride=1),
#             nn.BatchNorm2d(channel_in),
#             nn.ReLU(),
#             conv3x3(channel_in, channel_in, stride=1, groups=channel_in),
#             conv1x1(channel_in, channel_out, stride=1),
#             nn.BatchNorm2d(channel_out),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.op(x)


# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34
#     """
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()

#         self.residual_branch = nn.Sequential(
#             nn.Conv2d(in_channels,
#                       out_channels,
#                       kernel_size=3,
#                       stride=stride,
#                       padding=1,
#                       bias=False), nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels,
#                       out_channels * BasicBlock.expansion,
#                       kernel_size=3,
#                       padding=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion))

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels,
#                           out_channels * BasicBlock.expansion,
#                           kernel_size=1,
#                           stride=stride,
#                           bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion))

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_branch(x) + self.shortcut(x))


# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers
#     """
#     expansion = 4

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BottleNeck, self).__init__()
#         self.residual_branch = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels,
#                       out_channels,
#                       stride=stride,
#                       kernel_size=3,
#                       padding=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels,
#                       out_channels * BottleNeck.expansion,
#                       kernel_size=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels,
#                           out_channels * BottleNeck.expansion,
#                           stride=stride,
#                           kernel_size=1,
#                           bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion))

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_branch(x) + self.shortcut(x))


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class ResNet(nn.Module):
#     def __init__(self, block, layers, nclasses=100, args=None):
#         super(ResNet, self).__init__()
#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True))

#         self.stage2 = self._make_layer(block, 64, layers[0], 1)
#         self.stage3 = self._make_layer(block, 128, layers[1], 2)
#         self.stage4 = self._make_layer(block, 256, layers[2], 2)

#         self.expert_stages = nn.ModuleList([
#             self._make_expert_layer(block, 256, 512, layers[3], 2) for _ in range(args.nums_expert)
#         ])

#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.out_dim = 512 * block.expansion
#         self.nums_expert = args.nums_expert

#         self.proj_dim = args.proj_dim
#         if args.nums_expert == 6:
#             self.proj_dim = [128, 128, 128, 128, 128, 128]
#         elif args.nums_expert == 8:
#             self.proj_dim = [128, 128, 128, 128, 128, 128, 128, 128]

#         self.expert_proj_head = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.out_dim, self.out_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.out_dim, self.proj_dim[i])
#             ) for i in range(self.nums_expert)
#         ])

#         self.multi_heads = nn.ModuleList([
#             nn.ModuleList([]) for _ in range(self.nums_expert)
#         ])

#         self.args = args

#     @property
#     def n_params(self):
#         return sum(np.prod(p.size()) for p in self.parameters())

#     def _make_attention_layer(self, in_channels):
#         layers = [
#             DownConv(in_channels, in_channels),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid()
#         ]
#         return nn.Sequential(*layers)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block
#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer

#         Return:
#             return a resnet layer
#         """
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def _make_expert_layer(self, block, in_channels, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(in_channels, out_channels, stride))
#             in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def add_head(self, n_classes, bias=False):
#         for i in range(self.nums_expert):
#             self.multi_heads[i].append(nn.Linear(self.out_dim, n_classes, bias=bias).cuda())

#     def forward_general_features(self, x):
#         x0 = self.conv1(x)
#         x = self.stage2(x0)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         return x

#     def forward_expert_features(self, x, idx=None):
#         general_features = self.forward_general_features(x)
#         if idx is not None:
#             expert_feat = self.expert_stages[idx](general_features)
#             expert_feat = self.avg_pool(expert_feat)
#             expert_feat = expert_feat.view(expert_feat.size(0), -1)
#             return general_features, expert_feat
#         else:
#             expert_feat = []
#             for i in range(self.nums_expert):
#                 expert_feat_i = self.expert_stages[i](general_features)
#                 expert_feat_i = self.avg_pool(expert_feat_i)
#                 expert_feat_i = expert_feat_i.view(expert_feat_i.size(0), -1)
#                 expert_feat.append(expert_feat_i)
#             return expert_feat

#     def forward_expert_proj_head(self, x, idx=None):
#         if idx is not None:
#             return self.expert_proj_head[idx](x)
#         else:
#             return [self.expert_proj_head[i](x[i]) for i in range(self.nums_expert)]

#     def forward_expert_head(self, x, idx=None):
#         if idx is not None:
#             out = []
#             for head in self.multi_heads[idx]:
#                 out.append(head(x))
#             return torch.cat(out, dim=1)
#         else:
#             out = []
#             for i in range(self.nums_expert):
#                 cur_out = torch.cat([self.multi_heads[i][j](x[i]) for j in range(len(self.multi_heads[i]))], dim=1)
#                 out.append(cur_out)
#             return out

#     def features(self, x, idx=None):
#         return self.forward_expert_features(x, idx)

#     def head(self, x, use_proj=False):
#         if use_proj:
#             return self.proj_head(x)
#         else:
#             return torch.cat([head(x) for head in self.multi_heads], dim=1)

#     def forward(self, x, use_proj=False):
#         features = self.features(x)
#         outputs = self.head(features, use_proj)

#         if use_proj:
#             return features, outputs
#         else:
#             return outputs


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     # only load state_dict()
#     if pretrained:
#         pass
#     return model


# def resnet18_expert(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)


# def resnet34_expert(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet50_expert(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)


# def resnet101_expert(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
#                    progress, **kwargs)


# def resnet152_expert(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
#                    progress, **kwargs)

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_expert',
    'resnet34_expert',
    'resnet50_expert',
    'resnet101_expert',
    'resnet152_expert',
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
        return nn.ReLU(inplace=True)(self.residual_branch(x) + self.shortcut(x))


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
        return nn.ReLU(inplace=True)(self.residual_branch(x) + self.shortcut(x))


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


class ResNet(nn.Module):
    def __init__(self, block, layers, nclasses=100, args=None):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.stage4 = self._make_layer(block, 256, layers[2], 2)

        self.expert_stages = nn.ModuleList([
            self._make_expert_layer(block, 256, 512, layers[3], 2) for _ in range(args.nums_expert)
        ])

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.out_dim = 512 * block.expansion
        self.nums_expert = args.nums_expert

        self.proj_dim = args.proj_dim
        if args.nums_expert == 6:
            self.proj_dim = [128, 128, 128, 128, 128, 128]
        elif args.nums_expert == 8:
            self.proj_dim = [128, 128, 128, 128, 128, 128, 128, 128]

        self.expert_proj_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim, self.proj_dim[i])
            ) for i in range(self.nums_expert)
        ])

        self.multi_heads = nn.ModuleList([
            nn.ModuleList([]) for _ in range(self.nums_expert)
        ])

        self.args = args

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

    def _make_expert_layer(self, block, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride))
            in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def add_head(self, n_classes, bias=False):
        for i in range(self.nums_expert):
            self.multi_heads[i].append(nn.Linear(self.out_dim, n_classes, bias=bias).cuda())

    def forward_general_features(self, x):
        x0 = self.conv1(x)
        x = self.maxpool(x0)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward_expert_features(self, x, idx=None):
        general_features = self.forward_general_features(x)
        if idx is not None:
            expert_feat = self.expert_stages[idx](general_features)
            expert_feat = self.avg_pool(expert_feat)
            expert_feat = expert_feat.view(expert_feat.size(0), -1)
            return general_features, expert_feat
        else:
            expert_feat = []
            for i in range(self.nums_expert):
                expert_feat_i = self.expert_stages[i](general_features)
                expert_feat_i = self.avg_pool(expert_feat_i)
                expert_feat_i = expert_feat_i.view(expert_feat_i.size(0), -1)
                expert_feat.append(expert_feat_i)
            return expert_feat

    def forward_expert_proj_head(self, x, idx=None):
        if idx is not None:
            return self.expert_proj_head[idx](x)
        else:
            return [self.expert_proj_head[i](x[i]) for i in range(self.nums_expert)]

    def forward_expert_head(self, x, idx=None):
        if idx is not None:
            out = []
            for head in self.multi_heads[idx]:
                out.append(head(x))
            return torch.cat(out, dim=1)
        else:
            out = []
            for i in range(self.nums_expert):
                cur_out = torch.cat([self.multi_heads[i][j](x[i]) for j in range(len(self.multi_heads[i]))], dim=1)
                out.append(cur_out)
            return out

    def features(self, x, idx=None):
        return self.forward_expert_features(x, idx)

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


def resnet18_expert(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_expert(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_expert(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_expert(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152_expert(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)

