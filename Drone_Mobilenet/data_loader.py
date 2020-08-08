import torchvision
import torch
from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np


class train_transforms:
    def __init__(self):
        self.train_transform = Compose([
            Resize(224, 224, 3),
            Rotate(),
            # PadIfNeeded(min_height=70, min_width=70,),
            # RandomCrop(64, 64,),
            HorizontalFlip(),
            # CoarseDropout(max_holes=1, max_height=16, max_width=16),
            Normalize(mean=[0.3749, 0.4123, 0.4352], std=[0.3326, 0.3393, 0.3740]),
            ToTensor(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.train_transform(image=img)['image']
        return img


class test_transforms:
    def __init__(self):
        self.test_transform = Compose([
            Resize(224, 224, 3),
            Normalize(mean=[0.3749, 0.4123, 0.4352], std=[0.3326, 0.3393, 0.3740]),
            ToTensor(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.test_transform(image=img)['image']
        return img


def drone_dataloader():
    trainset = torchvision.datasets.ImageFolder(root="./drone_dataset/train/", transform=train_transforms())
    testset = torchvision.datasets.ImageFolder(root="./drone_dataset/val/", transform=test_transforms())

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    class_names = trainset.classes

    return trainloader, testloader, class_names, trainset
