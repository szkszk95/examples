import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from cyclegan.utils.dataset import DATASET
from itertools import chain

from cyclegan.Discriminator import Discriminator
from cyclegan.Generator import Generator
from cyclegan.dataloader import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--datasetA', help="images path of domain A",
                    default="/home/szk/PycharmProjects/open-reid/examples/data/viper/images")
parser.add_argument('--datasetB', help="images path of domain B",
                    default="/home/szk/PycharmProjects/open-reid/examples/data/cuhk03/images")
parser.add_argument('--dataroot', help='path to dataset',
                    default="./data")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.dataroot)
except OSError:
    pass

try:
    os.makedirs(opt.outf)
except OSError:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


dataset = dataloader(opt.dataroot, opt.datasetA, opt.datasetB, bs=opt.batchSize, if_generate=False)
datasetA = DATASET('/home/szk/PycharmProjects/open-reid/examples/data/cuhk03/images', 143, 128, 1)
datasetB = DATASET('/home/szk/PycharmProjects/open-reid/examples/data/viper/images', 143, 128, 1)
loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderA = iter(loader_A)
loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                       batch_size=opt.batchSize,
                                       shuffle=True,
                                       num_workers=2)
loaderB = iter(loader_B)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netGAB = Generator(ngf=opt.ngf, input_nc=opt.nc, output_nc=opt.nc).to(device)
netGAB.apply(weights_init)

netGBA = Generator(ngf=opt.ngf, input_nc=opt.nc, output_nc=opt.nc).to(device)
netGBA.apply(weights_init)
print(netGAB)

netDA = Discriminator(ndf=opt.ndf, input_nc=opt.nc).to(device)
netDA.apply(weights_init)

netDB = Discriminator(ndf=opt.ndf, input_nc=opt.nc).to(device)
netDB.apply(weights_init)
print(netDA)

L1Loss = nn.L1Loss()
BCELoss = nn.BCELoss()

optimizerDA = optim.Adam(netDA.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
optimizerDB = optim.Adam(netDB.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
optimizerG = optim.Adam(chain(netGAB.parameters(), netGBA.parameters()), lr=0.0002, betas=(0.5, 0.999))

label_real = 1
label_fake = 0

for epoch in range(opt.niter):
    for i in range(dataset.iters()):
        # imageA, imageB = dataset.next()
        # imageA, imageB = torch.Tensor(imageA).to(device), torch.Tensor(imageB).to(device)
        loaderA, loaderB = iter(loader_A), iter(loader_B)
        imageA = loaderA.next().to(device)
        imageB = loaderB.next().to(device)

        # --------train fDx---------
        # train DA,DB with real image
        netDA.zero_grad()
        netDB.zero_grad()

        out_real_A = netDA.forward(imageA)
        out_real_B = netDA.forward(imageB)
        label_r = torch.full(out_real_A.shape, label_real, device=device)

        errDA_real = BCELoss(out_real_A, label_r)
        errDB_real = BCELoss(out_real_B, label_r)

        err_real = errDA_real + errDB_real
        err_real.backward()

        # train DA,DB with fake image
        fake_AB = netGAB(imageA)
        fake_BA = netGBA(imageB)

        out_fake_A = netDB.forward(fake_AB.detach())
        out_fake_B = netDA.forward(fake_BA.detach())
        label_f = torch.full(out_fake_A.shape, label_fake, device=device)

        errDA_fake = BCELoss(out_fake_A, label_f)
        errDB_fake = BCELoss(out_fake_B, label_f)
        err_fake = errDA_fake + errDB_fake
        err_fake.backward()

        optimizerDA.step()
        optimizerDB.step()

        # --------train fGx---------
        # train GBA with fake image, L1loss between fake and real image
        for j in range(5):
            netGAB.zero_grad()
            netGBA.zero_grad()

            fake_AB = netGAB(imageA)
            fake_BA = netGBA(imageB)

            fake_ABA = netGBA(fake_AB)
            fake_BAB = netGAB(fake_BA)
            out_fake_ABA = netDA(fake_ABA)
            out_fake_BAB = netDB(fake_BAB)

            # GAN LOSS
            err_ABA = BCELoss(out_fake_ABA, label_r)
            err_BAB = BCELoss(out_fake_BAB, label_r)
            err_GAN = err_ABA+err_BAB

            # MSE LOSS
            dis_A = L1Loss(fake_ABA, imageA)
            dis_B = L1Loss(fake_BAB, imageB)
            err_MSE = dis_A + dis_B

            err_G = err_GAN+err_MSE
            err_G.backward()

            optimizerG.step()

        print('[{}/{}][{}/{}] netD_real {:.4f} netD_fake {:.4f} netG {:.4f}'.format(
            epoch,
            opt.niter,
            i,
            dataset.iters(),
            err_fake.data[0], err_real.data[0], err_G.data[0]))

        if i % 10 == 0:
            vutils.save_image(imageA, '{}/epoch_{}_real_A.png'.format(opt.outf, epoch), normalize=True)
            vutils.save_image(imageB, '{}/epoch_{}_real_B.png'.format(opt.outf, epoch), normalize=True)
            vutils.save_image(fake_AB, '{}/epoch_{}_fake_A2B.png'.format(opt.outf, epoch), normalize=True)
            vutils.save_image(fake_BA, '{}/epoch_{}_fake_B2A.png'.format(opt.outf, epoch), normalize=True)
            vutils.save_image(fake_ABA, '{}/epoch_{}_fake_A2B2A.png'.format(opt.outf, epoch), normalize=True)
            vutils.save_image(fake_BAB, '{}/epoch_{}_fake_B2A2B.png'.format(opt.outf, epoch), normalize=True)
