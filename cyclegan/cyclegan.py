import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nstd, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, nstd * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nstd * 8),
            nn.ReLU(True),
            # state size. (nstd*8) x 4 x 4
            nn.ConvTranspose2d(nstd * 8, nstd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nstd * 4),
            nn.ReLU(True),
            # state size. (nstd*4) x 8 x 8
            nn.ConvTranspose2d(nstd * 4, nstd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nstd * 2),
            nn.ReLU(True),
            # state size. (nstd*2) x 16 x 16
            nn.ConvTranspose2d(nstd * 2,     nstd, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nstd),
            nn.ReLU(True),
            # state size. (nstd) x 32 x 32
            nn.ConvTranspose2d(    nstd,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)