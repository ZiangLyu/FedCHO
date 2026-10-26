import torch
from torch import nn
import torchvision.models as models

def get_resnet18_for_cifar100(pretrained=False):


    model = models.resnet18(pretrained=pretrained)


    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


    model.maxpool = nn.Identity()


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)

    return model