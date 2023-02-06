import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# data
input_size = 784
hidden_layer = 512
second_hidden_layer = 256
output_size = 10
batch_size = 64

# dataset class contains data within
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# num of epochs
num_epochs = 10

# define our model
layer_1 = nn.Linear(input_size, hidden_layer)
layer_2 = nn.Linear(hidden_layer, second_hidden_layer)
layer_3 = nn.Linear(second_hidden_layer, output_size)


def predict(x, layer_1, layer_2, layer_3):
    layer_1_result = torch.relu(layer_1(x))
    layer_2_result = torch.relu(layer_2(layer_1_result))
    y_hat = torch.relu(layer_3(layer_2_result))
    return y_hat


# optimizer
# concatenated list of weights and biases across all layers
parameters = list(layer_1.parameters()) + list(layer_2.parameters()) + list(layer_3.parameters())
optimizer = optim.SGD(parameters, lr=0.01)

# define loss function
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs):
    for batch_id, (x,y) in enumerate(train_loader):
        #convert data to a vector
        x = x.view(-1, 784).float()

        #zero gradients
        optimizer.zero_grad()

        #predict output
        y_hat = predict(x, layer_1, layer_2, layer_3)

        #calc error
        loss = loss_fn(y_hat,y)

        #calculate the gradients
        loss.backward()

        #update the parameters as a result of backprop
        optimizer.step()

        #print status
        if batch_id % 200 == 0:
            print(f"{batch_id} / {len(train_loader)}")

    #test epoch
    correct = 0
    total_count = 0

    #dont need any gradients during testing
    #this is purely to test the classifier so far
    #no back prop is needed
    with torch.no_grad():
        for x, y in test_loader:

            # convert data to a vector
            x = x.view(-1, 784).float()

            # predict output
            y_hat = predict(x, layer_1, layer_2, layer_3)

            #check correctness
            _, pred_label = torch.max(y_hat.data, 1)
            total_count += x.data.size()[0]
            correct += (pred_label == y.data).sum()

    print(f"Correct: {(correct / total_count) * 100.} %")