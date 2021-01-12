import os
from tqdm import tqdm
import argparse
import torch
import utils
from torch.utils import data
import torch.nn as nn
import numpy as np
import random
from dataset import *
import network

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='../datasets/data', help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False, help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3, help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'], help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--num_workers", type=int, default=2, help='number of workers loading the data')
    
    parser.add_argument("--ckpt", default=None, type=str,help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100, help="epoch interval for eval (default: 100)")
    
    args = parser.parse_args()
    
    if args.dataset.lower() == 'voc':
        args.num_classes = 21
    elif args.dataset.lower() == 'cityscapes':
        args.num_classes = 19
      
    # If image has different size we cannot use batch > 1  
    if args.dataset=='voc' and not args.crop_val:
        args.val_batch_size = 1

    return args

def mkdirs():
    utils.mkdir('checkpoints')
    utils.mkdir('results')

def validate(opts, model, loader, device, metrics):
    model.eval()
    metrics.reset()
    
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0  

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            
            if opts.save_val_results:
                for i in range(len(images)):
                    utils.save_images(loader, images[i], targets[i], preds[i], denorm, img_id)
                    img_id += 1
            break
                        
        score = metrics.get_results()
    return score

def main():
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
        
    
    opts = get_argparser()
    mkdirs()
    
    # Setup CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    
    # Set up metrics
    metrics = utils.StreamSegMetrics(opts.num_classes)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    
    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    
    # Set up optimizer and criterion
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    
    # Load from checkpoint
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_itrs = checkpoint["cur_itrs"]
        best_score = checkpoint['best_score']
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory 
        
    model = nn.DataParallel(model)
    model.to(device)
    
    # =====  Train  =====
    
    while True: 
        model.train()
        cur_epochs += 1
        for (images, labels) in tqdm(train_loader):
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d_%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                print("validation...")
                model.eval()
                val_score = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d_%d.pth' %
                              (opts.model, opts.dataset,opts.output_stride, opts.random_seed))

                model.train()
            scheduler.step()
            
            if cur_itrs >= opts.total_itrs:
                return
            
    
    

if __name__ == '__main__':
    main()