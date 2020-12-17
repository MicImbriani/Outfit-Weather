import network
import dataset
import torch.nn.functional as F



# Create instance of Network class
network = network.Network()

# Create train_set imported from dataset file
train_set = dataset.train_set
image, label = dataset.get_sample_image()

# Unsqueeze the image to add a 4th dimension to turn a 3d image into a 1d batch
b = image.unsqueeze(0)

# Prediction
pred = network(b)

# Print raw results
print("The correct category is: ", label)
print("The prediction values for each category are:\n", pred)
print()

print("The probability values of the predictions are:\n", F.softmax(pred,dim=1))
print()

print(pred.argmax(dim=1))