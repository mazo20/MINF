import os
from tqdm import tqdm
from torchsummary import summary
import argparse
import torch
import utils
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from dataset import *
import network
from datetime import date

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='../datasets/data', help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--results_root", type=str, default='results', help="path to output the results and checkpoint")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='v3plus_resnet50',choices=['v3_resnet50',  'v3plus_resnet50',
                                 'v3_resnet101', 'v3plus_resnet101', 'v3_mobilenet', 'v3plus_mobilenet'], help='model name')
    parser.add_argument("--teacher_model", type=str, default='v3plus_resnet50',choices=['v3_resnet50',  'v3plus_resnet50',
                                 'v3_resnet101', 'v3plus_resnet101', 'v3_mobilenet', 'v3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,help="apply separable conv to decoder and aspp")
    parser.add_argument("--separable", default=None, choices=[None, 'bottleneck', 'grouped'])
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False, help="save segmentation results to \"./results\"")
    parser.add_argument("--total_epochs",     type=int,   default=30,               help="Number of epochs per training")
    parser.add_argument("--max_epochs",       type=int,   default=30)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'], help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--num_workers", type=int, default=4, help='number of workers loading the data')
    parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha for KD')
    parser.add_argument('--beta', default=1e3, type=float, help='beta for AT')
    parser.add_argument('--aux_loss', default='AT', type=str, help='AT or SE loss')
    
    parser.add_argument("--ckpt", default=None, type=str,help="restore from checkpoint")
    parser.add_argument("--teacher_ckpt", default=None, type=str,help="restore teacher from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student"], help="training mode")

    parser.add_argument("--loss_type", type=str, default='cross_entropy',choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--index",  type=int,   default=1)
    parser.add_argument("--count_flops", action="store_true", default=False)
    parser.add_argument("--score_interval", type=int, default=1, help="number of iterations for score printint")
    
    opts = parser.parse_args()
    
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    return opts

def mkdirs(opts):
    utils.mkdir('checkpoints')
    utils.mkdir('results')
    utils.mkdir(opts.results_root)

def validate(model, optimizer, scheduler, best_score, cur_epochs):
    
    model.eval()
    metrics.reset()
    
    if opts.save_val_results:
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0  

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs, ints = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            
            
            if opts.save_val_results:
                for i in range(len(images)):
                    at_maps = [ints[j][i] for j in range(len(ints))]
                    utils.save_images(val_loader, images[i], targets[i], preds[i], at_maps, denorm, img_id)
                    img_id += 1 
                        
        score = metrics.get_results()
        
    model.train()
    return score

def main():
    # Set up model
    model_map = {
        'v3_resnet50': network.deeplabv3_resnet50,
        'v3plus_resnet50': network.deeplabv3plus_resnet50,
        'v3_resnet101': network.deeplabv3_resnet101,
        'v3plus_resnet101': network.deeplabv3plus_resnet101,
        'v3_mobilenet': network.deeplabv3_mobilenet,
        'v3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    
    best_score = 0.0
    epoch      = 0
    
    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    teacher = None
    if opts.separable is not None: #and 'plus' in opts.model:
        # print(opts.separable)
        network.deeplab.convert_to_separable_conv(model.classifier, opts.separable == 'bottleneck')
        # network.deeplab.convert_to_separable_conv(model.backbone)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    macs, params = utils.count_flops(model, opts)
    if (opts.count_flops):
        return
    utils.create_result(opts, macs, params)
    
    # Set up optimizer and criterion
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    scheduler = utils.PolyLR(optimizer, opts.total_epochs * len(train_loader), power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    
    
    # Load from checkpoint
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        epoch = checkpoint.get("epoch", 0)
        best_score = checkpoint.get('best_score', 0.0)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory 
    else:
        model = nn.DataParallel(model)
        model.to(device)
        
    if opts.save_val_results:
        score = validate(model, optimizer, scheduler, best_score, cur_epochs)
        print(metrics.to_str(score)) 
        return
    
    if opts.mode == "student":
        teacher = model_map[opts.teacher_model](num_classes=opts.num_classes, output_stride=16)
            
        checkpoint = torch.load(opts.teacher_ckpt, map_location=torch.device('cpu'))
        teacher.load_state_dict(checkpoint["model_state"])
        teacher = nn.DataParallel(teacher)
        teacher.to(device)
        for param in teacher.parameters():
            param.requires_grad = False
    
    # =====  Train  =====
    
    for epoch in tqdm(range(epoch, opts.total_epochs)):
        
        if opts.mode == "teacher":
            train_teacher(model, optimizer, criterion, scheduler)
        else:
            train_student(model, teacher, optimizer, scheduler)
        
        score = validate(model, optimizer, scheduler, best_score, cur_epochs)
        print(metrics.to_str(score))
        utils.save_result(score, opts)
        
        utils.save_ckpt(opts.data_root.replace('/input', '') + '/output', opts, model, optimizer, scheduler, best_score, cur_epochs)   
        if score['Mean IoU'] > best_score or (opts.max_epochs != opts.total_epochs and epoch+1 == opts.total_epochs):
            best_score = score['Mean IoU']
            utils.save_ckpt(opts.data_root, opts, model, optimizer, scheduler, best_score, epoch+1) 
        

def train_teacher(net, optimizer, criterion, scheduler):
    net.train()
    metrics.reset()
    pbar = tqdm(train_loader)
    for images, labels in pbar:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        
        outputs, _ = net(images)
        loss = criterion(outputs, labels)
        
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(targets, preds)
        score = metrics.get_results()
        pbar.set_postfix({"IoU": score["Mean IoU"]})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
            
def train_student(net, teacher, optimizer, scheduler):
    net.train()
    pbar = tqdm(train_loader)
    for images, labels in pbar:
        images = Variable(images.to(device, dtype=torch.float32))
        labels = Variable(labels.to(device, dtype=torch.long))
        
        outputs_student, ints_student = net(images)
        outputs_teacher, ints_teacher = teacher(images)
        
        loss = utils.distillation(outputs_student, outputs_teacher, labels, opts.temperature, opts.alpha)
        
        at_teacher, at_student = utils.match_at_layers(ints_teacher, ints_student)  
        
        adjusted_beta = (opts.beta*3)/len(at_student)    
            
        for i in range(len(at_student)):        
            loss += adjusted_beta * utils.at_loss(at_student[i], at_teacher[i])
        
        preds = outputs_student.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(targets, preds)
        score = metrics.get_results()
        pbar.set_postfix({"IoU": score["Mean IoU"]})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    

if __name__ == '__main__':
    opts = get_argparser()
    opts.date = date.today()
    mkdirs(opts)
    
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
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    
    # Set up metrics
    metrics = utils.StreamSegMetrics(opts.num_classes)
    
    main()
    
    # python code/main.py  --data_root /disk/scratch/s1762992/deeplab/datasets/data/input --crop_val --batch_size 16  --gpu_id 0,1,2,3 --model deeplabv3_mobilenet --crop_size 128 --output_stride 32 --mode student --ckpt checkpoints/best_deeplabv3_mobilenet_voc_os16_2.pth
