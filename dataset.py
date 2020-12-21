import torch
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



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
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()]),
    #loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), num_workers=1),
    #data = next(iter(loader)),
    #print(data[0].mean(), data[0].std())
)



# Now I will wrap (LOAD into) the training set into 
# PyTorch's DataLoader object. This will give us access to
# the data in our desired format.
def get_train_loader(train_set, size, shuffle):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=size, shuffle=shuffle)
    return train_loader



# Get one sample image
def get_sample_image():
    sample = next(iter(train_set))
    image, label = sample
    return image, label
    
# Show one sample image
def show_sample_image(image, label):
    print(label)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.pause(10)



# Get sample batch 
def get_sample_batch(train_loader):
    batch = next(iter(train_loader))
    images, labels = batch
    return images, labels

# Show sammple batch
def show_sample_batch(images, labels):
    grid = tv.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15,15))
    print(labels)
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.pause(10)



