import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg
import math

__all__ = ['TransLab']


def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


@MODEL_REGISTRY.register(name='TransLab')
class TransLab(SegBaseModel):
    def __init__(self):
        super(TransLab, self).__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        else:
            c1_channels = 256
            c4_channels = 2048
            c2_channel = 512

        self.head = _DeepLabHead_attention(self.nclass, c1_channels=c1_channels, c4_channels=c4_channels, c2_channel=c2_channel)
        self.head_b = _DeepLabHead(1, c1_channels=c1_channels, c4_channels=c4_channels)

        self.fus_head1 = FusHead()
        self.fus_head2 = FusHead(inplane=2048)
        self.fus_head3 = FusHead(inplane=512)

        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)
        outputs = list()
        outputs_b = list()

        x_b = self.head_b(c4, c1)

        #attention c1 c4
        attention_map = x_b.sigmoid()

        c1 = self.fus_head1(c1, attention_map)
        c4 = self.fus_head2(c4, attention_map)
        c2 = self.fus_head3(c2, attention_map)

        x = self.head(c4, c2, c1, attention_map)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x_b = F.interpolate(x_b, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        outputs_b.append(x_b)#.sigmoid())

        return tuple(outputs), tuple(outputs_b)

    def evaluate(self, image):
        """evaluating network with inputs and targets"""
        scales = cfg.TEST.SCALES
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = torch.zeros((batch, self.nclass, h, w)).to(image.device)
        scores = None
        scores_boundary = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            outputs, outputs_boundary = self.forward(cur_img)
            outputs = outputs[0][..., :height, :width]
            outputs_boundary = outputs_boundary[0][..., :height, :width]

            score = _resize_image(outputs, h, w)
            score_boundary = _resize_image(outputs_boundary, h, w)

            if scores is None:
                scores = score
                scores_boundary = score_boundary
            else:
                scores += score
                scores_boundary += score_boundary
        return scores, scores_boundary


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.use_aspp = True
        self.use_decoder = True
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)
        if self.use_decoder:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            cat_fmap = torch.cat([x, c1], dim=1)
            return self.block(cat_fmap)

        return self.block(x)


class _DeepLabHead_attention(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, c2_channel=512, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead_attention, self).__init__()
        # self.use_aspp = cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP
        # self.use_decoder = cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER
        self.use_aspp = True
        self.use_decoder = True
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48

            self.c2_block = _ConvBNReLU(c2_channel, 24, 1, norm_layer=norm_layer)
            last_channels += 24

        self.block = nn.Sequential(
            SeparableConv2d(256+24+48, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

        self.block_c2 = nn.Sequential(
            SeparableConv2d(256+24, 256+24, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256+24, 256+24, 3, norm_layer=norm_layer, relu_first=False))


        self.fus_head_c2 = FusHead(inplane=256+24)
        self.fus_head_c1 = FusHead(inplane=256+24+48)


    def forward(self, x, c2, c1, attention_map):
        c1_size = c1.size()[2:]
        c2_size = c2.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)


        if self.use_decoder:
            x = F.interpolate(x, c2_size, mode='bilinear', align_corners=True)
            c2 = self.c2_block(c2)
            x = torch.cat([x, c2], dim=1)
            x = self.fus_head_c2(x, attention_map)
            x = self.block_c2(x)

            x = F.interpolate(x, c1_size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            x = torch.cat([x, c1], dim=1)
            x = self.fus_head_c1(x, attention_map)
            return self.block(x)

        return self.block(x)


class FusHead(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, inplane=256):
        super(FusHead, self).__init__()
        self.conv1 = SeparableConv2d(inplane*2, inplane, 3, norm_layer=norm_layer, relu_first=False)
        self.fc1 = nn.Conv2d(inplane, inplane // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(inplane // 16, inplane, kernel_size=1)

    def forward(self, c, att_map):
        if c.size() != att_map.size():
            att_map = F.interpolate(att_map, c.size()[2:], mode='bilinear', align_corners=True)

        atted_c = c * att_map
        x = torch.cat([c, atted_c], 1)#512
        x = self.conv1(x) #256

        weight = F.avg_pool2d(x, x.size(2))
        weight = F.relu(self.fc1(weight))
        weight = torch.sigmoid(self.fc2(weight))
        x = x * weight
        return x
