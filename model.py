import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def swish(x):
    return x * torch.sigmoid(x)


class FeatureExtractor(nn.Module):
    def __init__(self, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        cnn = torchvision.models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))
        x = self.conv9(x)

        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upscale_factor=2, n_filters=64, inplace=False):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upscale_factor
        self.conv1 = nn.Conv2d(3, n_filters, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), ResidualBlock(n_filters, 3, n_filters, 1))

        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(n_filters, n_filters))
        self.conv3 = nn.Conv2d(n_filters, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))
        y = x.clone()

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)


class UpSampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=2):
        super(UpSampleConvLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        y = self.upsample_layer(x)
        y = self.reflection_pad(y)
        y = self.conv(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, k=3, n=64, s=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = swish(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + x
        return y


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, 1, padding=1)
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.up_layer = UpSampleConvLayer(in_channels, out_channels, 3, 1)
        # self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        # y = self.up_layer(x)
        # y = self.conv(x)
        y = self.convT(x)
        # y = self.shuffler(y)
        y = self.bn(y)
        y = swish(y)
        return y
