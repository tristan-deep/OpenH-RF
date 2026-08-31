#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }
 
class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    
# (B, 40, 1000, 161) -> (B, 1, 401, 161)
class InversionNet(nn.Module):
    def __init__(self, enc_ch=[6, 6, 6, 7, 7, 8, 8, 9], enc_side=[1, 0, 0, 0, 0, 0], 
                 bottle_conv=(8, 6), bottle_deconv=(7, 3), dec_ch=[9, 8, 7, 6, 5, 4, 3],
                 crop=[-15, -16, -23, -24], **kwargs):
        super(InversionNet, self).__init__()
        assert len(enc_ch) == len(enc_side) + 2 # 2 for first and last layer of encoder
        enc_ch = [2**c for c in enc_ch]
        dec_ch = [2**c for c in dec_ch]
        self.crop = crop
        bottle_conv = tuple(bottle_conv)
        bottle_deconv = tuple(bottle_deconv)

        layers = [ConvBlock(40, enc_ch[0], kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))]
        for i in range(1, len(enc_ch) - 1):
            if enc_side[i - 1]:
                layers.append(ConvBlock(enc_ch[i-1], enc_ch[i], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)))
                layers.append(ConvBlock(enc_ch[i], enc_ch[i], kernel_size=(3, 1), padding=(1, 0)))
            else:
                layers.append(ConvBlock(enc_ch[i-1], enc_ch[i], stride=2))
                layers.append(ConvBlock(enc_ch[i], enc_ch[i]))
        layers.append(ConvBlock(enc_ch[-2], enc_ch[-1], kernel_size=bottle_conv, padding=0))
        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(len(dec_ch)):
            if i == 0:
                layers.append(DeconvBlock(enc_ch[-1], dec_ch[i], kernel_size=bottle_deconv))
            else:
                layers.append(DeconvBlock(dec_ch[i-1], dec_ch[i], kernel_size=4, stride=2, padding=1))
            layers.append(ConvBlock(dec_ch[i], dec_ch[i]))
        self.decoder = nn.Sequential(*layers)
        self.output = ConvBlock_Tanh(dec_ch[-1], 1)

    def forward(self, x):
        # Encoder Part
        # 500, 250, 125,  63,  32,  16,   8
        # 161, 161,  81,  41,  21,  11,   6
        #  64,  64,  64, 128, 128, 256, 256
        # Decoder Part
        # 7 * 2**6 = 448, 3 * 2**6 = 192
        # crop = [-15, -16, -23, -24]
        # output: 401, 161
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.pad(x, self.crop, mode="constant", value=0)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = InversionNet() # 20447515
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' % total_params)
    ckpt = torch.load('InversionNet_weights_only.pth', map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        if missing:
            print('Missing keys: ', missing)
        if unexpected:
            print('Unexpected keys: ', unexpected)
    else:
        print('All keys matched successfully.')
    x= torch.rand((1, 40, 1000, 161))
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(y.shape)
