import torch
from torch import nn
from torch.nn import functional as F

from .network_utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], opts=None):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate, opts)

        self.classifier = nn.Sequential(
            AtrousSeparableConvolution(304, 256, 3, padding=1, bias=False, opts=opts),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature, activations = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) ), activations
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36], opts=None):
        super(DeepLabHead, self).__init__()
        
        self.aspp = ASPP(in_channels, aspp_dilate, opts)

        self.classifier = nn.Sequential(
            AtrousSeparableConvolution(256, 256, 3, padding=1, bias=False, opts=opts),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        out, activations = self.aspp(feature['out'])
        return self.classifier(out), activations
        

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True, opts=None):
        super(AtrousSeparableConvolution, self).__init__()
        
        if opts.separable == 'grouped':
            self.body = nn.Sequential(
                # Separable Conv
                nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
                # PointWise Conv
                nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            )
        elif opts.separable == 'bottleneck':
            bottleneck = out_channels // 4
            self.body = nn.Sequential(
                # Bottleneck
                nn.Conv2d(in_channels, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias),
                nn.BatchNorm2d(bottleneck),
                # Separable Conv
                nn.Conv2d( bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=bottleneck),
                nn.BatchNorm2d(bottleneck),
                # PointWise Conv
                nn.Conv2d( bottleneck, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            )
        else:
            '''
            No separable convolution
            '''
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            )
                
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, opts):
        modules = [
            AtrousSeparableConvolution(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if opts.large_aspp == 'medium':
            modules = [
                AtrousSeparableConvolution(in_channels, 2*out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(2*out_channels),
                AtrousSeparableConvolution(2*out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        elif opts.large_aspp == 'large':
            modules = [
                AtrousSeparableConvolution(in_channels, 4*out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(4*out_channels),
                AtrousSeparableConvolution(4*out_channels, 2*out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(2*out_channels),
                AtrousSeparableConvolution(2*out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        elif opts.large_aspp == 'extra_large':
            modules = [
                AtrousSeparableConvolution(in_channels, 4*out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(4*out_channels),
                AtrousSeparableConvolution(4*out_channels, 2*out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(2*out_channels),
                AtrousSeparableConvolution(2*out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(out_channels),
                AtrousSeparableConvolution(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False, opts=opts),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, opts):
        super(ASPP, self).__init__()
        self.opts = opts
        out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)),
            ASPPConv(in_channels, out_channels, rate1, opts),
            ASPPConv(in_channels, out_channels, rate2, opts),
            ASPPConv(in_channels, out_channels, rate3, opts),
            ASPPPooling(in_channels, out_channels)
        ]
        
        if opts.kernel_sharing == 'true':
            shared_weights = []
            
            for child in modules[1][0].children():
                for m in child.children():
                    shared_weights.append(m.weight)
                    if opts.only_3_kernel_sharing == 'true':
                        break
                
            for child in modules[2][0].children():
                for i, m in enumerate(child.children()):
                    m.weight = shared_weights[i]
                    if opts.only_3_kernel_sharing == 'true':
                        break
                    
            for child in modules[3][0].children():
                for i, m in enumerate(child.children()):
                    m.weight = shared_weights[i]
                    if opts.only_3_kernel_sharing == 'true':
                        break
            
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        activations = []
        for conv in self.convs:
            out = conv(x)
            res.append(out)
            if self.opts.at_type == 'aspp-atrous' or self.opts.at_type == 'aspp-all':
                activations.append(out)
                
        res = torch.cat(res, dim=1)
        out = self.project(res)
        if self.opts.at_type == 'aspp-output' or self.opts.at_type == 'aspp-all':
            activations.append(out)
            
        return out, activations



def convert_to_separable_conv(module, bottleneck=False):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias,
                                      bottleneck)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child, bottleneck))
    return new_module