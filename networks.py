import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net):
    print('Initializing...')
    net.apply(weights_init_normal)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.BatchNorm2d(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.BatchNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = []
        model += [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                  nn.BatchNorm2d(ngf),
                  nn.ReLU(True)]
        model += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(ngf * 2),
                  nn.ReLU(True)]
        model += [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(ngf * 4),
                  nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 4)]

        model += [nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.BatchNorm2d(ngf * 2),
                  nn.ReLU(True)]
        model += [nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.BatchNorm2d(ngf),
                  nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()

        model = []
        model += [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(ndf * 2),
                  nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(ndf * 4),
                  nn.LeakyReLU(0.2, True)]
        model += [nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
                  nn.BatchNorm2d(ndf * 8),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #x = torch.cat([x, label], 1)
        return self.model(x)

# GANLoss
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        #self.loss = nn.MSELoss()
        #self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)