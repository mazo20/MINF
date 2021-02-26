from ptflops import get_model_complexity_info


def count_flops(model, opts):
    macs, params = get_model_complexity_info(model, (3, opts.crop_size, opts.crop_size), as_strings=True,
                                           print_per_layer_stat=opts.count_flops, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))    
    
    return macs, params