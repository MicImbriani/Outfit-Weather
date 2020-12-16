import torch
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from fastai.vision import *


# Data manipulation consists of 3 steps:
# 1) EXTRACTION of the raw data
# 2) TRANSFORMATION into tensor 
# 3) LOADING into an object



# Takes the Fashion-MNIST dataset and downloads it in a new 
# directory. The dataset won't be downloaded a second time 
# if it is already present.
# The First 4 lines EXTRACT the dataset from the web.
# The last line TRANSFORMS the raw image data is transformed 
# into a tensor using PyTorch's ToTensor() object.
train_set = tv.datasets.FashionMNIST(
    root = '.\data\FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)


# Now I will wrap (LOAD into) the training set into 
# PyTorch's DataLoader object. This will give us access to
# the data in our desired format.

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

# To visualise one datapoint i.e. one image

sample = next(iter(train_set))
image, label = sample

#plt.imshow(image.squeeze(),cmap='gray')
#plt.pause(100)



# To visualise multiple images from one batch

batch = next(iter(train_loader))
images, labels = batch

grid = tv.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15,15))
#plt.imshow(np.transpose(grid,(1,2,0)))
#plt.pause(5)



# Check whether any of the images imported is corrupted 

corrupted = verify_images(images)
print(corrupted)