import argparse
import network
import dataset
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from itertools import product


def print_results(label, pred):
    """Prints raw results.

    Args:
        label ([tensor]): [Label for each of the item predicted]
        pred ([tensor]): [Prediction made for each item]
    Returns:
        None
    """    
    print("The correct category is: ", label)
    print("The prediction values for each category are:\n", pred)
    print()

    print("The probability values of the predictions are:\n", F.softmax(pred, dim=1))
    print()

    print(pred.argmax(dim=1))



# Compute and print the total number of correct predictions
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()



# Compute and print all predictions 
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch 

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    
    return all_preds



def confusion_matrix(ts, tp):
    conf_mat = confusion_matrix(ts.targets, tp.argmax(dim=1))

    names = ('T-shirt/top', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    
    plt.figure(figsize=(10,10))
    plt.title('Confusion Matrix')
    plt.labels('names')

    table = plt.table(conf_mat,
                    rowLabels=names,
                    colLabels=names,
                    loc='bottom')
    plt.show()
    return


############################################################################################################



# Create instance of Network class
network = network.Network()

# Create training set object importing from dataset file
train_set = dataset.train_set



# I will use the product module to substitute the for loop in the training process.
# To do that, I need to create a dictionary with all the parameters I have.
parameters = dict(
    lr = [0.01, 0.001],
    batch_size = [10, 100, 1000],
    shuffle = [True, False])

param_values = [v for v in parameters.values()]


# Iterate over the several cartesian products of the 3 parameters.
for lr, batch_size, shuffle in product(*param_values):

    # Create train_set and train_loader imported from dataset file
    train_loader = dataset.get_train_loader(train_set, batch_size, shuffle)
    images, labels = dataset.get_sample_batch()

    # Create the optimizer 
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Unsqueeze the image to add a 4th dimension to turn a 3d image into a 1d batch
    #b = image.unsqueeze(0)
    #image, label = dataset.get_sample_image()



    # Create TensorTable instance and add images and graph
    comment = f'batch_size={batch_size} lr={learning_rate} shuffle={shuffle}'
    tb = SummaryWriter(comment=comment)
    grid = tv.utils.make_grid(images)
    tb.add_image('images', grid)
    tb.add_graph(network, images)



    for epoch in range(1):
        # Counters for visualisation of progress during training
        total_loss = 0
        total_correct = 0

        # For loop for training the network
        for batch in train_loader:
            # Get batches 
            images, labels = batch

            # Pass batch to make prediction 
            preds = network(images)
            #get_num_correct(preds, labels)

            # Calculate loss
            loss = F.cross_entropy(preds, labels)
            #print(loss.item())

            # Clear the gradients 
            optimizer.zero_grad()

            # Calculate gradients 
            loss.backward()

            # Update weights
            optimizer.step()



            # Update the couters. Multiply by batch_size to keep track of which batch size produced which loss.
            total_loss += loss.item() * batch_size
            total_correct += get_num_correct(preds, labels)

            # Print some information to keep track of the progress 
            print("epoch:", epoch, "total correct:", total_correct, "loss:", total_loss)
        
        tab.add_scalar('Loss', total_loss, epoch)
        tab.add_scalar('Number Correct', total_correct, epoch)
        tab.add_scalar('Accuracy', total_correct/len(train_set), epoch)

        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print(total_correct/len(train_set))





# Turn off gradient tracking locally to remove overhead of keeping track of gradients/creating graph
with torch.no_grad():
    prediction_loader = dataset.train_loader
    train_preds = get_all_preds(network, prediction_loader)

#confusion_matrix(train_set, train_preds)


tb.close()


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--image_path", required=True, help="Path to the image to be used.")
args = vars(ap.parse_args())

my_img = args["image_path"]

transform = transforms.ToTensor()
my_img = transform(my_img)
new_pred = network(my_img)
print(new_pred)