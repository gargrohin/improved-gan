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

import torchvision.datasets as dset
import torchvision.transforms as transforms
# from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from model import get_inception_score

# DCGAN
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

nc = 3
ngf = 64
z_dim = 100
ndf = 64
G = generator().to(device)

path_G = "../../GANCF/models/cifar10_dcgan_1/dc64_ganns_200_G.pth"
G.load_state_dict(torch.load(path_G))
G.eval()

image_gan = []

z_dim = 100
batch_size = 200
for i in range(25):
    z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
    img = G(z)
    if i == 0:
        images = img
    else:
        images = torch.cat((images, img), dim = 0)

images = images.cpu().numpy()
print(images.shape)
for x in images:
    images_gan.append(x)


# cifar = dset.CIFAR10(root='../datasets/cifar10_data/', download=True,
#                             transform=transforms.Compose([
#                                 transforms.Scale(32),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                             ])
# )

# images = cifar.data
# images_list = []
# for x in images:
#     images_list.append(x)
print("\nCalculating IS...\n")
print(get_inception_score(images_gan))
