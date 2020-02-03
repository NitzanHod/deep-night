import torch
import torch.nn as nn
import torch.nn.functional as func


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        x2 = func.pad(x2, (diff_x // 2, int(diff_x / 2),
                           diff_y // 2, int(diff_y / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height,
                                                                                                s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UNet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.inc = InConv(in_ch, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = OutConv(32, out_ch)
        self.depth2space = DepthToSpace(2)

    def forward(self, x):
        #         print(x.size())
        x1 = self.inc(x)
        #         print(x1.size())
        x2 = self.down1(x1)
        #         print(x2.size())
        x3 = self.down2(x2)
        #         print(x3.size())
        x4 = self.down3(x3)
        #         print(x4.size())
        x5 = self.down4(x4)
        #         print(x5.size())
        x = self.up1(x5, x4)
        #         print(x.size())
        x = self.up2(x, x3)
        #         print(x.size())
        x = self.up3(x, x2)
        #         print(x.size())
        x = self.up4(x, x1)
        #         print(x.size())
        x = self.outc(x)
        #         print(x.size())
        x = self.depth2space(x)
        #         print(x.size())
        return x