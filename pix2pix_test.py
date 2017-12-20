import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
#from model import Generator
from networks import ResnetGenerator
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=1024, help='input size')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = './data_256/valA/'
save_dir = './test_results/'
model_dir = './model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
'''
# Data pre-processing
test_transform = transforms.Compose([transforms.Scale(params.input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
'''

from dataloader import DataLoader
transform = transforms.Compose([transforms.Scale(256),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

test_data = DataLoader('./data_256/valA', './data_256/valB', transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)

# Load model
G = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=6)
#G.cuda()
G.load_state_dict(torch.load(model_dir + 'generator_param.pkl'))

# Test
for i, (input, target) in enumerate(test_data_loader):
    # input & target image data
    x_ = Variable(input)#.cuda())
    y_ = Variable(target)#.cuda())

    gen_image = G(x_)
    gen_image = gen_image.data

    # Show result for test data
    utils.plot_test_result(input, target, gen_image, i, training=False, save=True, save_dir=save_dir)

    print('%d images are generated.' % (i + 1))

