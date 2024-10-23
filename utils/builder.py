import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from randaugment import CIFAR10Policy, ImageNetPolicy, Cutout, RandAugment
from data.noisy_cifar import NoisyCIFAR10, NoisyCIFAR100,NoisyImageNet
import numpy as np
import cv2

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


  
def build_transform(rescale_size=512, crop_size=448):

    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    cifar_train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])    


    tiny_train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(64),
        # torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    tiny_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    tiny_train_transform_strong_aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(64),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.4, hue=0.07),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])  




    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform, 'train_strong_aug': train_transform_strong_aug,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform, 'cifar_train_strong_aug': cifar_train_transform_strong_aug,
            'tiny_train': tiny_train_transform, 'tiny_test': tiny_test_transform, 'tiny_train_strong_aug': tiny_train_transform_strong_aug}
            


def build_cifar10n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR10(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=False)   

    test_data = NoisyCIFAR10(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=False)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=True, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=True, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


def build_tiny_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyImageNet(root="/home/jxr/proj/PLLNL/tiny-imagenet-200", train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyImageNet(root="/home/jxr/proj/PLLNL/tiny-imagenet-200", train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}

