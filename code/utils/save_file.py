from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch.nn.functional as F

def save_images(loader, image, target, pred, attention_maps, denorm, img_id, root):
    root = root + '/images'
    if not os.path.exists(root):
        os.mkdir(root)
        
    image = image.detach().cpu().numpy()
    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
    target = loader.dataset.decode_target(target).astype(np.uint8)
    pred = loader.dataset.decode_target(pred).astype(np.uint8)

    Image.fromarray(image).save( '%s/%d_image.png'  % (root, img_id))
    Image.fromarray(target).save('%s/%d_target.png' % (root, img_id))
    Image.fromarray(pred).save(  '%s/%d_pred.png'   % (root, img_id))

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(pred, alpha=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('%s/%d_overlay.png' % (root, img_id), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    # for i in range(len(attention_maps)):
    #     fig = plt.figure()
    #     plt.imshow(image)
    #     plt.axis('off')
        
    #     at_map = F.interpolate(attention_maps[i].pow(2).mean(0).unsqueeze(0).unsqueeze(0), size=image.shape[:2], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        
    #     plt.imshow(at_map, interpolation='bicubic', alpha=0.7)
    #     ax = plt.gca()
    #     ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    #     ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    #     plt.savefig('%s/%d_at_%d.png' % (root, img_id, i), bbox_inches='tight', pad_inches=0)
    #     plt.close()
    
def save_at_map(map):
    plt.imshow(map, interpolation='bicubic')
    plt.savefig('results/test.png', bbox_inches='tight')
    plt.close()

    
def create_result(opts, macs, params):
    path = f'{opts.results_root}/{opts.mode}_{opts.model}_os_{opts.output_stride}_{opts.crop_size}_{opts.index}.csv'
    
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',)
            spamwriter.writerow(['model='+opts.model, 'teacher='+opts.teacher_model, 
                                'separable='+str(opts.separable_conv), 'os='+str(opts.output_stride), 
                                'crop='+str(opts.crop_size), 'MAdds='+str(macs), 'params='+str(params)])
            spamwriter.writerow(['Overall_Acc', 'Mean_Acc', 'FreqW_Acc', 'Mean_IoU'])
    
def save_result(score, opts):
    path = f'{opts.results_root}/{opts.mode}_{opts.model}_os_{opts.output_stride}_{opts.crop_size}_{opts.index}.csv'
    with open(path, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',)
        spamwriter.writerow([score['Overall Acc'], score['Mean Acc'], score['FreqW Acc'], score['Mean IoU']])
