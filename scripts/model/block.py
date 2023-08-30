import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.bn is not None:
            y = self.relu(self.bn(self.conv(x)))
        else:
            y = self.relu(self.conv(x))
        return y

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm = True):
        super(DoubleConvBlock, self).__init__()
        self.dconvb1 = ConvBlock(in_channels, out_channels, batch_norm)
        self.dconvb2 = ConvBlock(out_channels, out_channels, batch_norm)
    def forward(self, x):
        y = self.dconvb2(self.dconvb1(x))
        return y
    
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UpConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_x):
        y = self.up(x)
        y = torch.cat([y, skip_x], dim=1)
        y = self.conv(y)
        return y
