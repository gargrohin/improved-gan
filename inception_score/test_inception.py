import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

# from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from model import get_inception_score

import torchvision.datasets as dset
import torchvision.transforms as transforms

cifar = dset.CIFAR10(root='../datasets/cifar10_data/', download=True,
                            transform=transforms.Compose([
                                transforms.Scale(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
)

images = cifar.data

print(model(images))