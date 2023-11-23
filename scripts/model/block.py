import torch
import torch.nn as nn

class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.bn is not None:
            return self.relu(self.bn(self.conv(x)))
        return self.relu(self.conv(x))

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(DoubleConvBlock, self).__init__()
        self.dconvb1 = SingleConvBlock(in_channels, out_channels, kernel_size, stride, padding, batch_norm)
        self.dconvb2 = SingleConvBlock(out_channels, out_channels, kernel_size, stride, padding, batch_norm)

    def forward(self, x):
        return self.dconvb2(self.dconvb1(x))
    
class UpSingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, up_mode='transpose'):
        super(UpSingleConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConvBlock(in_channels, out_channels) 

    def forward(self, x, skip_x, custom_padding=False):
        x = self.up(x)
        
        if custom_padding:
            d_x, d_y = skip_x.size()[3] - x.size()[3], skip_x.size()[2] - x.size()[2] # Input is CHW
            x = nn.functional.pad(x, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        y = torch.cat([skip_x, x], dim=1)
        return self.conv(y)

class UpDoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, kernel_size=2, stride=2, padding=0, up_mode='transpose'):
        super(UpDoubleConvBlock, self).__init__()
        if up_mode == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvBlock(in_channels, out_channels) 

    def forward(self, x, skip_x, custom_padding=False):
        x = self.up(x)
        
        if custom_padding:
            d_x, d_y = skip_x.size()[3] - x.size()[3], skip_x.size()[2] - x.size()[2] # Input is CHW
            x = nn.functional.pad(x, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        y = torch.cat([skip_x, x], dim=1)
        return self.conv(y)
