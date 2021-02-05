from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def save_images(loader, image, target, pred, denorm, img_id):
    image = image.detach().cpu().numpy()
    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
    target = loader.dataset.decode_target(target).astype(np.uint8)
    pred = loader.dataset.decode_target(pred).astype(np.uint8)

    Image.fromarray(image).save('results/%d_image.png' % img_id)
    Image.fromarray(target).save('results/%d_target.png' % img_id)
    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(pred, alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def create_result(opts):
    path = 'results/%s_%s_os_%d_%s_%d' % (opts.mode, opts.model, opts.output_stride, opts.date, opts.random_seed)
    with open(path + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',)
        spamwriter.writerow(['model='+opts.model, 'teacher='+opts.teacher_model, 
                             'separable='+str(opts.separable_conv), 'os='+str(opts.output_stride), 
                             'crop='+str(opts.crop_size)])
        spamwriter.writerow(['Overall_Acc', 'Mean_Acc', 'FreqW_Acc', 'Mean_IoU'])
    
def save_result(score, opts):
    path = 'results/%s_%s_os_%d_%s_%d' % (opts.mode, opts.model, opts.output_stride, opts.date, opts.random_seed)
    with open(path + '.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',)
        spamwriter.writerow([score['Overall Acc'], score['Mean Acc'], score['FreqW Acc'], score['Mean IoU']])
