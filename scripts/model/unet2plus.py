import torch
import torch.nn as nn

from .block import DoubleConvBlock, UpDoubleConvBlock

# U-Net++ model - To be updated
class UNet2Plus(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UNet2Plus, self).__init__()

        # Joint pool layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder pathway
        self.encoder1 = DoubleConvBlock(input_dim, 64)
        self.encoder2 = DoubleConvBlock(64, 128)
        self.encoder3 = DoubleConvBlock(128, 256)
        self.encoder4 = DoubleConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(512, 1024)

        # Decoder pathway
        self.decoder1 = UpDoubleConvBlock(1024, 512)
        self.decoder2 = UpDoubleConvBlock(512, 256)
        self.decoder3 = UpDoubleConvBlock(256, 128)
        self.decoder4 = UpDoubleConvBlock(128, 64)
    
        # Output layer
        self.outconv = nn.Conv2d(64, output_dim, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec1 = self.decoder1(bottleneck, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)

        return self.outconv(dec4)
