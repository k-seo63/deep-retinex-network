import torch
import torch.nn as nn


class UNet_retinex(nn.Module):
    """ Network outputs reflectance, shading and wb-vec
    """
    class double_conv(nn.Module):
        '''(conv => BN => ReLU) * 2'''
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet_retinex.double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                activation()
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    class inconv(nn.Module):
        def __init__(self, in_ch, out_ch, activation=nn.ReLU):
            super(UNet_retinex.inconv, self).__init__()
            self.conv = UNet_retinex.double_conv(in_ch, out_ch, activation)

        def forward(self, x):
            x = self.conv(x)
            return x

    class down(nn.Module):
        def __init__(self, in_ch, out_ch, down_conv_layer=nn.Conv2d, activation=nn.ReLU):
            super(UNet_retinex.down, self).__init__()
            self.mpconv = nn.Sequential(
                down_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                padding=1, kernel_size=(3, 3),
                                stride=(2, 2), bias=False),
                UNet_retinex.double_conv(in_ch, out_ch, activation=activation)
            )

        def forward(self, x):
            x = self.mpconv(x)
            return x

    class up(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, up_conv_layer=nn.ConvTranspose2d, activation=nn.ReLU):
            super(UNet_retinex.up, self).__init__()
            self.upconv = up_conv_layer(in_channels=in_ch, out_channels=in_ch,
                                        kernel_size=(4, 4), stride=(2, 2),
                                        padding=1, bias=False)
            self.conv = UNet_retinex.double_conv(mid_ch, out_ch, activation=activation)

        def forward(self, x1, x2):
            x1 = self.upconv(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    class outconv_relu(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet_retinex.outconv_relu, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.ReLU()
                )

        def forward(self, x):
            x = self.conv(x)
            return x

    class outconv_sigmoid(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(UNet_retinex.outconv_sigmoid, self).__init__()
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1),
                    nn.Sigmoid()
                )

        def forward(self, x):
            x = self.conv(x)
            return x
    
    class globalconv_a(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch, activation=nn.ReLU):
            super(UNet_retinex.globalconv_a, self).__init__()
            self.globalconv = nn.Sequential(
                    UNet_retinex.double_conv(in_ch, mid_ch, activation=activation),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(mid_ch, out_ch, 1),
                    nn.ReLU()
                )

        def forward(self, x):
            x = self.globalconv(x)
            return x

    def __init__(self, n_channels=3, n_classes=3,
                 up_conv_layer=nn.ConvTranspose2d,
                 down_conv_layer=nn.Conv2d,
                 activation=nn.ReLU):
        super(UNet_retinex, self).__init__()
        self.inc = self.inconv(n_channels, 64, activation=activation)
        self.down1 = self.down(64, 128, activation=activation)
        self.down2 = self.down(128, 256, activation=activation)
        self.down3 = self.down(256, 512, activation=activation)
        self.down4 = self.down(512, 512, activation=activation)

        self.up1_a = self.up(512, 1024, 256, activation=activation)
        self.up2_a = self.up(256, 512, 128, activation=activation)
        self.up3_a = self.up(128, 256, 64, activation=activation)
        self.up4_a = self.up(64, 128, 64, activation=activation)
        self.outc_a = self.outconv_relu(64, 1)

        self.gconv = self.globalconv_a(512, 64, 3)

        self.up1_b = self.up(512, 1024, 256, activation=activation)
        self.up2_b = self.up(256, 512, 128, activation=activation)
        self.up3_b = self.up(128, 256, 64, activation=activation)
        self.up4_b = self.up(64, 128, 64, activation=activation)
        self.outc_b = self.outconv_sigmoid(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_a = self.up1_a(x5, x4)
        x_a = self.up2_a(x_a, x3)
        x_a = self.up3_a(x_a, x2)
        x_a = self.up4_a(x_a, x1)
        x_a = self.outc_a(x_a)

        x_a_wb = self.gconv(x5)
        x_a = x_a * x_a_wb

        x_b = self.up1_b(x5, x4)
        x_b = self.up2_b(x_b, x3)
        x_b = self.up3_b(x_b, x2)
        x_b = self.up4_b(x_b, x1)
        x_b = self.outc_b(x_b)

        return x_a, x_b, x_a_wb
