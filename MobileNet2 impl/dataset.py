import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

def create_dataset(params):
    """
    Create datasets for training, testing and validating
    :return datasets: a python dictionary includes three datasets
                        datasets[
    """
    phase = ['train', 'val']
    # if params.dataset_root is not None and not os.path.exists(params.dataset_root):
    #     raise ValueError('Dataset not exists!')

    transform = {'train': transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              RandomHorizontalFlip(),
                                              ToTensor()
                                              ]),
                 'val'  : transforms.Compose([Rescale((params.image_size, params.image_size)),
                                              ToTensor()
                                              ]),
                 'test' : transforms.Compose([transforms.Resize(params.image_size),
                                              ToTensor()
                                              ])}

    # file_dir = {p: os.path.join(params.dataset_root, p) for p in phase}

    # datasets = {Cityscapes(file_dir[p], mode=p, transforms=transform[p]) for p in phase}
    datasets = {p: PASCALVOC(params.dataset_root, mode=p, transform=transform[p]) for p in phase}

    return datasets

class PASCALVOC(Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self, root, mode='train', transform=None, download=False):
        self.transform = transform
        self.train = mode
        self.root = os.path.join(root, 'VOCdevkit/VOC2012')
        file_list_dir = os.path.join(self.root, 'list')
        file_name = mode + '.txt'
        
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Unsupported mode %s' % mode)

        if download:
            self.download()
            
        self.images = []
        self.masks = []
        
        print(file_name)
        
        for line in open(os.path.join(file_list_dir, file_name)):
            image = self.root + line.split()[0]
            mask = self.root + line.split()[1]
            assert os.path.isfile(image)
            assert os.path.isfile(mask)
            self.images.append(image)
            self.masks.append(mask)
            
        ##TODO: REMOVE THIS LATER
        
        self.images = self.images[:100]
        self.masks = self.masks[:100]

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])[...,::-1]
        label = cv2.imread(self.masks[index])[...,::-1]
            
        sample = {'image': img, 'label': label[:,:,0]}

        if self.transform:
            sample = self.transform(sample)
            
        return sample


    def __len__(self):
        return len(self.images)

    def download(self):
        raise NotImplementedError('Automatic download not yet implemented.')

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['label'] = image, label

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, output_stride=16):
        self.output_stride = output_stride

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # reset label shape
        # w, h = label.shape[0]//self.output_stride, label.shape[1]//self.output_stride
        # label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # label[label == 255] = 19
        label = label.astype(np.int64)

        # normalize image
        image /= 255

        sample['image'], sample['label'] = torch.from_numpy(image), torch.from_numpy(label)

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __call__(self, sample, p=0.5):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) < p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        sample['image'], sample['label'] = image, label

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]

        label = label[top: top + new_h, left: left + new_w]

        sample['image'], sample['label'] = image, label

        return sample
