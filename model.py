import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Encoder (Contracting Path)
        self.down1 = self.double_conv(n_channels, 64)
        self.down2 = self.double_conv(64, 128)
        self.down3 = self.double_conv(128, 256)
        self.down4 = self.double_conv(256, 512)
        self.bottleneck = self.double_conv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        # Decoder (Expansive Path)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up4 = self.double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up3 = self.double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = self.double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=0),  # Chuẩn bài báo: no padding
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=0),
            nn.ReLU(inplace=True)
        )

    def crop_and_concat(self, bypass, upsampled):
        # Cắt bypass (encoder) cho khớp kích thước upsampled (decoder)
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = bypass[:, :, c:c + upsampled.size()[2], c:c + upsampled.size()[2]]
        return torch.cat((bypass, upsampled), 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))
        b  = self.bottleneck(self.pool(c4))

        d4 = self.conv_up4(self.crop_and_concat(c4, self.up4(b)))
        d3 = self.conv_up3(self.crop_and_concat(c3, self.up3(d4)))
        d2 = self.conv_up2(self.crop_and_concat(c2, self.up2(d3)))
        d1 = self.conv_up1(self.crop_and_concat(c1, self.up1(d2)))
        return self.out_conv(d1)
