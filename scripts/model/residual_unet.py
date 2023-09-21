import torch
import torch.nn as nn
from .block import DoubleConvBlock, ResidualConvBlock, ResidualInputBlock, UpResidualConvBlock 

# Residual U-Net model
class ResidualUNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualUNet, self).__init__()

        # Encoder pathway
        self.encoder1 = ResidualInputBlock(input_dim, 64)
        self.encoder2 = ResidualConvBlock(64, 128)
        self.encoder3 = ResidualConvBlock(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(256, 512)

        # Decoder pathway
        self.decoder1 = UpResidualConvBlock(512, 256)
        self.decoder2 = UpResidualConvBlock(256, 128)
        self.decoder3 = UpResidualConvBlock(128, 64)
        
        # Output layer
        self.outconv = nn.Conv2d(64, output_dim, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        bottleneck = self.bottleneck(enc3)
        
        dec1 = self.decoder1(bottleneck, enc3)
        dec2 = self.decoder2(dec1, enc2)
        dec3 = self.decoder3(dec2, enc1)

        return self.outconv(dec3)
