from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
