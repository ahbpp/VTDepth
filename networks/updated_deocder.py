# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class UpdatedDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 norm_layer=nn.BatchNorm2d, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 128, 160, 256])
        self.out_scales = [1, 2, 3, 4]

        self.num_layers = len(self.num_ch_enc) - 1
        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], self.num_output_channels,
                                                    kernel_size=3, padding=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        # print(len(input_features), self.num_layers)
        for i in range(self.num_layers, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                # print(x[0].shape, i)
                # print(input_features[i - 1].shape)
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if self.out_scales[i] in self.scales:
                self.outputs[("disp", self.out_scales[i])] = self.sigmoid(self.convs[("dispconv", i)](x))
        self.outputs[("disp", 0)] = self.last_layer(x)
        return self.outputs

class UpdatedDepthDecoderLighter2(UpdatedDepthDecoder):

    def __init__(self, num_ch_enc,
                 num_ch_dec=np.array([32, 64, 128, 256]),
                 reduce_last=False,
                 scales=range(4), num_output_channels=1, norm_layer=nn.BatchNorm2d, use_skips=True):
        super().__init__(num_ch_enc, scales, num_output_channels, norm_layer, use_skips)
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.out_scales = [1, 2, 3, 4]

        self.num_layers = len(self.num_ch_enc) - 1
        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Sequential(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                nn.ReLU(),
                norm_layer(num_ch_out)
            )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Sequential(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                nn.ReLU(),
                norm_layer(num_ch_out)
            )

        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], self.num_output_channels,
                                                    kernel_size=3, padding=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        if reduce_last:
            self.last_layer = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0] * 4, kernel_size=1),
                nn.PixelShuffle(2),
                nn.ReLU(),
                nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3, padding=1,
                          groups=self.num_ch_dec[0]),
                nn.ReLU(),
                nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.last_layer = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0] * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(),
                nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

class UpdatedDepthDecoderLighter3(UpdatedDepthDecoder):

    def __init__(self, num_ch_enc,
                 num_ch_dec=np.array([32, 64, 128, 256]),
                 scales=range(4), num_output_channels=1, norm_layer=nn.BatchNorm2d, use_skips=True):
        super().__init__(num_ch_enc, scales, num_output_channels, norm_layer, use_skips)
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.out_scales = [1, 2, 3, 4]

        self.num_layers = len(self.num_ch_enc) - 1
        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Sequential(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                nn.ReLU(),
                norm_layer(num_ch_out)
            )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Sequential(
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                nn.ReLU(),
                norm_layer(num_ch_out)
            )

        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], self.num_output_channels,
                                                    kernel_size=3, padding=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0] * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(self.num_ch_dec[0] // 2, self.num_ch_dec[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_ch_dec[0] // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


class UpdatedDepthDecoderLighter(nn.Module):
    def __init__(self, num_ch_enc,
                 num_ch_dec=np.array([32, 64, 128, 256]),
                 scales=range(4), num_output_channels=1,
                 norm_layer=nn.BatchNorm2d, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec
        self.out_scales = [1, 2, 3, 4]

        self.num_layers = len(self.num_ch_enc) - 1
        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

        for s in self.scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], self.num_output_channels,
                                                    kernel_size=3, padding=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], max(self.num_ch_dec[0] // 2, 32), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(max(self.num_ch_dec[0] // 2, 32), 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        # print(len(input_features), self.num_layers)
        for i in range(self.num_layers, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                # print(x[0].shape, i)
                # print(input_features[i - 1].shape)
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if self.out_scales[i] in self.scales:
                self.outputs[("disp", self.out_scales[i])] = self.sigmoid(self.convs[("dispconv", i)](x))
        x = upsample(x)
        self.outputs[("disp", 0)] = self.last_layer(x)
        return self.outputs

class UpdatedDepthDecoderv2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 norm_layer=nn.BatchNorm2d, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = list(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 128, 160, 256])
        self.out_scales = self.scales[1:]

        self.num_layers = len(self.num_ch_enc) - 1
        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layers, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_layers else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = nn.Sequential(
                                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1),
                                nn.ReLU(),
                                norm_layer(num_ch_out)
                                )

        for s in self.out_scales:
            self.convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s-1], self.num_output_channels,
                                                    kernel_size=3, padding=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(self.num_layers, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            # if i+1 in self.out_scales:
            self.outputs[("disp", self.out_scales[i])] = \
                    self.sigmoid(self.convs[("dispconv", self.out_scales[i])](x))
        self.outputs[("disp", 0)] = self.last_layer(x)
        return self.outputs