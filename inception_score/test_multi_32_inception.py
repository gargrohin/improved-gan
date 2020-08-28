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
# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(z_dim, ngf * 8, 2, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         # if input.is_cuda and self.ngpu > 1:
#         #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         # else:
#         output = self.main(input)
#         return output

class generator(nn.Module):
    def __init__(self, ch = 512, bw=4):
        super(generator, self).__init__()
        self.ch = ch
        self.bw = bw
        self.lin = nn.Linear(z_dim, 4*4*ch)
        self.bn0 = nn.BatchNorm1d(4*4*ch)
        self.main = nn.Sequential(

            nn.ConvTranspose2d(ch, ch//2, 4, 2, 1),
            nn.BatchNorm2d(ch//2),
            nn.ReLU(True),

            # state size. (ngf*8) x 2x2
            nn.ConvTranspose2d(ch//2, ch//4, 4, 2, 1),
            nn.BatchNorm2d(ch//4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4x4
            nn.ConvTranspose2d(ch//4, ch//8, 4, 2, 1),
            nn.BatchNorm2d(ch//8),
            nn.ReLU(True),
            # state size. (ngf*2) x 8x8
            nn.ConvTranspose2d(ch//8, 3, 3, 1, 1),
            # nn.BatchNorm2d(3),
            nn.Tanh(),
            # state size. (ngf) x 16x16
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            # nn.Tanh()
            # state size. (nc) x 32x32
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.bn0(self.lin(input)).view(-1,self.ch, self.bw, self.bw)
        output = self.main(output)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

nc = 3
ngf = 64
z_dim = 128
ndf = 64
G = generator().to(device)

scores = []

for gen_ind in range(20,81,20):
    path_G = "../../GANCF/models/cifar10_chainer_1/dc64_ganns_" + str(gen_ind) + "_G.pth"
    G.load_state_dict(torch.load(path_G))
    G.eval()

    images_gan = []

    z_dim = 100
    batch_size = 100
    with torch.no_grad():
        for i in range(250):
            z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
            img = G(z).cpu()
            if i == 0:
                images = img
            else:
                images = torch.cat((images, img), dim = 0)
    images = images.view(-1,32,32,3)
    images = images.detach().cpu().numpy()
    print(images.shape)
    images=np.round((images+1)*(255/2))

    torch.cuda.empty_cache()
    
    for x in images:
        images_gan.append(x)
    print("\nCalculating IS...\n")
    scores.append((get_inception_score(images_gan), gen_ind))
    print("\n----------------------------------\n")

print("cifar10 chainer 1 32")
print(scores)



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
