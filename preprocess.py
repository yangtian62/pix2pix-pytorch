import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib.image as mpimg
from scipy import misc

images = []
for img_path in glob.glob('./maps/train/*.jpg'):
    images.append(mpimg.imread(img_path))

for i, im in enumerate(images):
    im1 = im[:,:600,:]
    im1 = misc.imresize(im1,(128,128,3))
    im2 = im[:, 600:, :]
    im2 = misc.imresize(im2, (128, 128, 3))
    misc.imsave('./data_128/trainA/%d.jpg'%i, im1)
    misc.imsave('./data_128/trainB/%d.jpg'%i, im2)

