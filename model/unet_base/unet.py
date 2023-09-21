from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class double_conv(nn.Module):
    """
    两次卷积

    """

    def __init__(self, input_channels, output_channels):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """
    下采样
    """

    def __init__(self, input_channels, output_channels):
        super(down, self).__init__()

        self.up = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(input_channels, output_channels)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up_conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Unet(nn.Module):

    def __init__(self, input_channels=3, output_channels=1):
        super(Unet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.inc = double_conv(input_channels, filters[0])
        self.down1 = down(filters[0], filters[1])
        self.down2 = down(filters[1], filters[2])
        self.down3 = down(filters[2], filters[3])
        self.down4 = down(filters[3], filters[4])

        self.up_mid5 = up_conv(filters[4], filters[3])
        self.up5 = double_conv(filters[4], filters[3])

        self.up_mid4 = up_conv(filters[3], filters[2])
        self.up4 = double_conv(filters[3], filters[2])

        self.up_mid3 = up_conv(filters[2], filters[1])
        self.up3 = double_conv(filters[2], filters[1])

        self.up_mid2 = up_conv(filters[1], filters[0])
        self.up2 = double_conv(filters[1], filters[0])

        self.outc = nn.Conv2d(filters[0], output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        d5 = self.up_mid5(e5)
        d5 = torch.cat((d5, e4), dim=1)
        d5 = self.up5(d5)

        d4 = self.up_mid4(d5)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.up4(d4)

        d3 = self.up_mid3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.up3(d3)

        d2 = self.up_mid2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.up2(d2)

        out = self.outc(d2)

        return out