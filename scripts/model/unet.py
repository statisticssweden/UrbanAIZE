import torch
import torch.nn as nn
from .block import DoubleConvBlock

# Classic U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Joint pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder pathway
        self.encoder1 = DoubleConvBlock(in_channels, 64)
        self.encoder2 = DoubleConvBlock(64, 128)
        self.encoder3 = DoubleConvBlock(128, 256)
        self.encoder4 = DoubleConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(512, 1024)

        # Decoder pathway
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvBlock(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvBlock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvBlock(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((enc4, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((enc3, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((enc2, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        dec4 = self.upconv4(dec3)
        dec4 = torch.cat((enc1, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        output = self.outconv(dec4)
        return output
