""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utransformer_parts import *

class U_Transformer(nn.Module):
    def __init__(self, 
        in_chans: int,
        out_chans: int,
        chans: int = 32
        ):
        super().__init__()
        self.n_channels = in_chans
        self.n_classes = out_chans
        self.chans = chans

        self.inc = DoubleConv(in_chans, chans)
        self.down1 = Down(chans, chans*2)
        self.down2 = Down(chans*2, chans*4)
        self.down3 = Down(chans*4, chans*8)
        self.MHSA = MultiHeadSelfAttention(chans*8)
        self.up1 = TransformerUp(chans*8, chans*4)
        self.up2 = TransformerUp(chans*4, chans*2)
        self.up3 = TransformerUp(chans*2, chans)
        self.outc = OutConv(chans, out_chans)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)