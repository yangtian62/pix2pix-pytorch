import torch
from torchvision import transforms
from torch.autograd import Variable
from networks import ResnetGenerator, Discriminator, init_weights, GANLoss
import utils
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='map', help='input dataset')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--num_epochs', type=int, default=1, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = './data_256/'
save_dir = './results/'
model_dir = './model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

from dataloader import DataLoader
transform = transforms.Compose([transforms.Scale(256),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_data = DataLoader('./data_256/trainA', './data_256/trainB', transform=transform)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=1,
                                                shuffle=True)

test_data = DataLoader('./data_256/valA', './data_256/valB', transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()



# Models
G = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=6)
D = Discriminator(input_nc=6, ndf=64)
#G.cuda()
#D.cuda()
init_weights(G)
init_weights(D)
#G.init_weights(mean=0.0, std=0.02)
#D.init_weights(mean=0.0, std=0.02)

# Loss function
#BCE_loss = torch.nn.BCELoss()#.cuda()
BCE_loss = GANLoss()
L1_loss = torch.nn.L1Loss()#.cuda()


# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

D_losses = []
G_losses = []
#step = 0
print('Start training!')
for epoch in range(params.num_epochs):
    #D_losses = []
    #G_losses = []

    # training
    for i, (input, target) in enumerate(train_data_loader):

        # input & target image data
        x_ = Variable(input)
        y_ = Variable(target)

        # Train discriminator with real data
        #D_real_decision = D(x_, y_).squeeze()
        #real_ = Variable(torch.ones(D_real_decision.size()))#.cuda())
        #D_real_loss = BCE_loss(D_real_decision, real_)
        # Real
        real_AB = torch.cat((x_, y_), 1)
        pred_real = D(real_AB)
        D_real_loss = BCE_loss(pred_real, True)


        # Train discriminator with fake data
        #gen_image = G(x_)
        #D_fake_decision = D(x_, gen_image).squeeze()
        #fake_ = Variable(torch.zeros(D_fake_decision.size()))#.cuda())
        #D_fake_loss = BCE_loss(D_fake_decision, fake_)
        # Fake
        fake_B = G(x_)
        fake_AB = torch.cat((x_, fake_B), 1)
        pred_fake = D(fake_AB.detach())
        D_fake_loss = BCE_loss(pred_fake, False)


        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        # Train generator
        #gen_image = G(x_)
        #D_fake_decision = D(x_, gen_image).squeeze()
        #G_fake_loss = BCE_loss(D_fake_decision, real_)
        G_fake_loss = BCE_loss(pred_fake, True)
        l1_loss = L1_loss(fake_B, y_) * 10 #self.opt.lambda_A: default = 10

        # L1 loss
        #l1_loss = params.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.data[0])
        G_losses.append(G_loss.data[0])

        print('Epoch [%d/%d], Iter [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.data[0], G_loss.data[0]))

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    gen_image = G(Variable(test_input))#.cuda()))
    gen_image = gen_image.cpu().data
    utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)

np.savetxt("D_losses.csv", D_losses, delimiter=",")
np.savetxt("G_losses.csv", G_losses, delimiter=",")
# Plot average losses
utils.plot_loss(D_losses, G_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
utils.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
