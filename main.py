import os
import argparse
from config import Params
from dataset import create_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from network import SegmentationNetwork


LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')


def main():
    # add argumentation
    parser = argparse.ArgumentParser(description='Segmentation network')
    parser.add_argument('--root', default='./data/cityscapes', help='Path to your dataset')
    args = parser.parse_args()
    params = Params()

    # parse args
    if not os.path.exists(args.root):
        if not os.path.exists(params.dataset_root):
            raise ValueError('ERROR: Root %s not exists!' % args.root)
    else:
        params.dataset_root = args.root

    # create dataset and transformation
    LOG('Creating Dataset and Transformation......')
    dataset = create_dataset(params)
    LOG('Creation Succeed.\n')
    
    # create model
    LOG('Initializing MobileNet and DeepLab......')
    net = SegmentationNetwork(params, dataset)
    LOG('Model Built.\n')

    # let's start to train!
    net.Train()
    # net.Test()

if __name__ == '__main__':
    main()