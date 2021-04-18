""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

NORM_FUNC = nn.InstanceNorm2d
ACTIVATION_FUNC = nn.LeakyReLU(negative_slope=0.2, inplace=True) #nn.ReLU(inplace=True)
POOL_FUNC = nn.MaxPool2d(2)

#====================================================================
#=========================U-Net======================================
#====================================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            NORM_FUNC(mid_channels),
            ACTIVATION_FUNC,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            NORM_FUNC(out_channels),
            ACTIVATION_FUNC
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            POOL_FUNC,
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)    

    def forward(self, x):
        return self.conv(x)

#====================================================================
#==========================Transformer===============================
#====================================================================
class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super(MultiHeadDense, self).__init__()
        #self.weight = nn.Parameter(torch.Tensor(d, d))
        self.linear = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)

        #TODO, verify this
        #x = F.linear(x, self.weight)
        x = self.linear(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
    
    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        device = torch.device("cuda:0")
        pe = torch.zeros(d_model, height, width, device=device)

        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2, device=device) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width, device=device).unsqueeze(1)
        pos_h = torch.arange(0., height, device=device).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel)
        self.key = MultiHeadDense(channel)
        self.value = MultiHeadDense(channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.positional_encoding_2d(c, h, w)
        x = x + pe
        x = x.reshape(b, c, h*w).permute(0, 2, 1) #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))#[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x

class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            POOL_FUNC,
            nn.Conv2d(channelS, channelS, kernel_size=1),
            NORM_FUNC(channelS),
            ACTIVATION_FUNC
        )
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            NORM_FUNC(channelS),
            ACTIVATION_FUNC
        )
        self.query = MultiHeadDense(channelS)
        self.key = MultiHeadDense(channelS)
        self.value = MultiHeadDense(channelS)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            NORM_FUNC(channelS),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channelS, channelS, kernel_size=2, stride=2)
        )
        self.Yconv2 = nn.Sequential(
            nn.ConvTranspose2d(channelY, channelY, kernel_size=2, stride=2),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            NORM_FUNC(channelS),
            ACTIVATION_FUNC
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        V = self.value(S1)
        Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z



class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels, Schannels, kernel_size=3, stride=1, padding=1),
            NORM_FUNC(Schannels),
            ACTIVATION_FUNC,
            nn.Conv2d(Schannels, Schannels, kernel_size=3, stride=1, padding=1),
            NORM_FUNC(Schannels),
            ACTIVATION_FUNC
        )

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x

