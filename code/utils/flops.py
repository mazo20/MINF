from thop import profile, clever_format
import torch

def count_flops(model, crop_size):
    input = torch.randn(1, 3, crop_size, crop_size)
    macs, params = profile(model, inputs=(input, ))
    return int(macs), int(params), clever_format([macs, params], "%.3f")