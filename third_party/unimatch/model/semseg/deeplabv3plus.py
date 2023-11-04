import third_party.unimatch.model.backbone.resnet as resnet
from third_party.unimatch.model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, 
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False, only_fp=False):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if only_fp:
            out_fp = self._decode(nn.Dropout2d(0.5)(c1),
                                nn.Dropout2d(0.5)(c4))
            out_fp = F.interpolate(out_fp, size=(h, w), mode="bilinear", align_corners=True)
            return out_fp
        elif need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate, norm=nn.BatchNorm2d, act=nn.ReLU):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          norm(out_channels),
                          act(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm2d, act=nn.ReLU):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm(out_channels),
                                 act(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=None, bn=True, relu=True):
        super(ASPPModule, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates
        norm = nn.BatchNorm2d if bn else nn.Identity
        act = nn.ReLU if relu else nn.Identity

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm(out_channels),
                                act(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm, act)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm, act)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm, act)
        self.b4 = ASPPPooling(in_channels, out_channels, norm, act)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     norm(out_channels),
                                     act(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
