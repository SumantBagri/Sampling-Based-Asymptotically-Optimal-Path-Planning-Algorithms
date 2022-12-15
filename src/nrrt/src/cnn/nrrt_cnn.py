#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, f_lc=False, channel_reduce=False, downsample=True, downsample_stride=1):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        
        if channel_reduce:
            out_channels_1 = out_channels // 4
            stride_1 = 2
        else:
            out_channels_1 = in_channels
            stride_1 = 1


        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels_1,
                               kernel_size=1,
                               stride=stride_1,
                               padding=0,
                               dtype=torch.float32)

        # downsampling required for low level feature map in NRRTCNN
        if f_lc:
            self.conv1 = nn.Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=1,
                                     stride=2,
                                     padding=0,
                                     dtype=torch.float32)

        self.batch_norm1 = nn.BatchNorm2d(out_channels_1, dtype=torch.float32)

        self.conv2 = nn.Conv2d(in_channels=out_channels_1,
                               out_channels=out_channels_1,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               dtype=torch.float32)
        self.batch_norm2 = nn.BatchNorm2d(out_channels_1, dtype=torch.float32)

        self.conv3 = nn.Conv2d(in_channels=out_channels_1,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1, 
                               padding=0,
                               dtype=torch.float32)
            
        self.batch_norm3 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)

        # define downsampling for skip connection
        if self.downsample:
            self.identity_downsample = nn.Sequential(    
                nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=downsample_stride, 
                            padding=0,
                            dtype=torch.float32),
                nn.BatchNorm2d(out_channels, dtype=torch.float32)
            )
        
    def forward(self, x):
        identity = x.clone()

        # ResNet Block
        x = self.elu(self.batch_norm1(self.conv1(x)))
        x = self.elu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        x= self.dropout(x)
        
        # downsample original dimensions
        if self.downsample:
            identity = self.identity_downsample(identity)
        
        # skip connection
        x += identity
        x = self.elu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, num_channels=1, f_lc=False):
        super(ResNet, self).__init__()
        modules = []

        # conv1 layer from ResNet
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, 
                      out_channels=64, 
                      kernel_size=7, 
                      stride=2,
                      padding=3,
                      dtype=torch.float32),
            nn.BatchNorm2d(64, dtype=torch.float32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1)
        )

        downsample_stride = 2 if f_lc else 1

        # add first bottleneck layer
        conv2 = ResNetBlock(in_channels=64, 
                            out_channels=256,
                            f_lc=f_lc,
                            downsample_stride=downsample_stride)
        modules.append(conv2)

        # add remaining layers for high level feature map
        if not f_lc:
            modules.append(ResNetBlock(in_channels=256, out_channels=512, channel_reduce=True, downsample_stride=2))
            modules.append(ResNetBlock(in_channels=512, out_channels=1024, channel_reduce=True, downsample_stride=2))
            modules.append(ResNetBlock(in_channels=1024, out_channels=2048, channel_reduce=True, downsample_stride=2))

        self.module = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.module(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aconv1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                dilation=2,
                                dtype=torch.float32)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.2)

        self.aconv2 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=4,
                                dilation=4,
                                dtype=torch.float32)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.aconv3 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=8,
                                dilation=8,
                                dtype=torch.float32)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.2)

        self.aconv4 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=12,
                                dilation=12,
                                dtype=torch.float32)
        self.batch_norm4 = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout(0.2)

    def forward(self, x):
        x1 = self.dropout1(self.elu1(self.batch_norm1(self.aconv1(x))))
        x2 = self.dropout2(self.elu2(self.batch_norm2(self.aconv2(x))))
        x3 = self.dropout3(self.elu3(self.batch_norm3(self.aconv3(x))))
        x4 = self.dropout4(self.elu4(self.batch_norm4(self.aconv4(x))))

        return torch.concat((x1, x2, x3, x4), dim=1)                                

class NRRTCNN(nn.Module):
    def __init__(self, clearance, stepsize=1.0):
        super(NRRTCNN, self).__init__()
        self.clearance = clearance
        self.stepsize = stepsize

        # feature maps
        self.feature_map_low = ResNet(f_lc=True)
        self.feature_map_high = ResNet(f_lc=False)

        # robotic attributes layers
        self.attribute_low = nn.Conv2d(in_channels=2, 
                                       out_channels=32,
                                       kernel_size=1,
                                       dtype=torch.float32)

        self.attribute_high = nn.Conv2d(in_channels=2, 
                                        out_channels=64,
                                        kernel_size=1, 
                                        dtype=torch.float32)

        # ASPP layer
        self.aspp = ASPP(2048, 64)
        self.batch_norm1 = nn.BatchNorm2d(256, dtype=torch.float32)
        self.elu1 = nn.ELU()

        # decoding block 1
        self.decode1 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=608,
                                               out_channels=304,
                                               kernel_size=2,
                                               stride=2,
                                               dtype=torch.float32),
                            nn.Conv2d(in_channels=304,
                                      out_channels=304,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(304, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                            nn.Conv2d(in_channels=304,
                                      out_channels=304,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(304, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                        )
        
        # decoding block 2
        self.decode2 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=304,
                                               out_channels=152,
                                               kernel_size=2,
                                               stride=2,
                                               dtype=torch.float32),
                            nn.Conv2d(in_channels=152,
                                      out_channels=152,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(152, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                            nn.Conv2d(in_channels=152,
                                      out_channels=152,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(152, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                        )
        
        # decoding block 3
        self.decode3 = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=152,
                                               out_channels=76,
                                               kernel_size=2,
                                               stride=2,
                                               dtype=torch.float32),
                            nn.Conv2d(in_channels=76,
                                      out_channels=76,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(76, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                            nn.Conv2d(in_channels=76,
                                      out_channels=76,
                                      kernel_size=3,
                                      padding=1,
                                      dtype=torch.float32),
                            nn.BatchNorm2d(76, dtype=torch.float32),
                            nn.ELU(),
                            nn.Dropout(0.2),
                        )

        # final layers
        self.conv1 = nn.Conv2d(in_channels=76, 
                               out_channels=1, 
                               kernel_size=1, 
                               dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        dev = torch.device("cuda")
        attribute_tensor = torch.tensor([self.clearance, self.stepsize], dtype=torch.float32, requires_grad=True).reshape(1, 2, 1, 1)
        attribute_tensor = attribute_tensor.expand(x.shape[0], 2, 1, 1)
        attribute_tensor = attribute_tensor.to(dev)

        # low level attributes
        ra_l = self.attribute_low(attribute_tensor.clone())
        #ra_l = einops.repeat(self.ra_l, 'n c w h -> n c (w2 w) (h2 h)', w2=x.shape[1] // 8, h2=x.shape[2] // 2)
        ra_l = ra_l.expand(ra_l.shape[0], ra_l.shape[1], x.shape[2] // 8, x.shape[3] // 8)

        # high level attributes
        ra_h = self.attribute_high(attribute_tensor.clone())
        #ra_h = einops.repeat(ra_h, 'n c w h -> n c (w2 w) (h2 h)', w2=(x.shape[1] // 32), h2=(x.shape[2] // 32))
        ra_h = ra_h.expand(ra_h.shape[0], ra_h.shape[1], x.shape[2] // 32, x.shape[3] // 32)
        
        # low level featuremap
        fm_l = self.feature_map_low(x.clone())
        
        self.fm_l = fm_l
        self.ra_l = ra_l
        
        fm_l = torch.concat((fm_l, ra_l), 1)
        
        # high level featuremap
        fm_h = self.feature_map_high(x.clone())

        # aspp
        fm_h = self.elu1(self.batch_norm1(self.aspp(fm_h)))
        fm_h = torch.concat((fm_h, ra_h), 1)

        # resize high level featuremap
        fm_h = F.interpolate(fm_h, [fm_h.shape[2]*4, fm_h.shape[3]*4], mode='bilinear', align_corners=True)
        
        # concat featuremaps
        fm_t = torch.concat((fm_h, fm_l), 1)

        # decoding blocks
        fm_t = self.decode1(fm_t)
        fm_t = self.decode2(fm_t)
        fm_t = self.decode3(fm_t)

        # final conv layer + softmax
        fm_t = self.conv1(fm_t)
        self.out = fm_t
        fm_t = self.sigmoid(fm_t)

        return fm_t


    
    