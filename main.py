import network
import dataset
import torch.nn.functional as F

import torch.optim as optim



# Print raw results
def print_results(label, pred):
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
        


############################################################################################################



# Create instance of Network class
network = network.Network()

# Create train_set imported from dataset file
train_set = dataset.train_set
image, label = dataset.get_sample_image()
images, labels = dataset.get_sample_batch()

# Create the optimizer 
optimizer = optim.Adam(network.parameters(), lr=0.01)

# Unsqueeze the image to add a 4th dimension to turn a 3d image into a 1d batch
b = image.unsqueeze(0)



for epoch in range(5):
    # Counters for visualisation of progress during training
    total_loss = 0
    total_correct = 0

    # For loop for training the network
    for batch in dataset.train_loader:
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



        # Update the couters 
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

        # Print some information to keep track of the progress 
        print("epoch:", epoch, "total correct:", total_correct, "loss:", total_loss)

print(total_correct/len(train_set))