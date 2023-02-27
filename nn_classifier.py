from typing import *

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim


class nnClassifier:

    def __init__(self, layers: List[int], epochs: int):
        self.layers = layers
        self.batch_size = 64
        self.epochs = epochs
        self.linear_layers = []
        self.accuracy = []

        for i in range(1, len(layers)):
            self.linear_layers.append(nn.Linear(layers[i - 1], layers[i]))

    def predict(self, x, linear_layers):
        for layer in linear_layers:
            x = layer(x)
        return x

    def train_and_test(self):
        # dataset class contains data within
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        parameters = [list(x.parameters()) for x in self.linear_layers]
        optimizer = optim.SGD(parameters[0], lr=0.05)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            for batch_id, (x, y) in enumerate(train_loader):
                # convert data to a vector - but still with a size of 64 (batch size), 784 as the nn.Linear function can take a batched input
                x = x.view(-1, 784).float()

                # zero gradients
                optimizer.zero_grad()

                # predict output
                y_hat = self.predict(x, self.linear_layers)

                # calc error
                loss = loss_fn(y_hat, y)
                print(loss)

                # calculate the gradients
                loss.backward()

                # update the parameters as a result of backprop
                optimizer.step()

                # print status
                #if batch_id % 200 == 0:
                    #print(f"{batch_id} / {len(train_loader)}")

            # test epoch
            correct = 0
            total_count = 0

            # don't need any gradients during testing
            # this is purely to test the classifier so far
            # no back prop is needed
            with torch.no_grad():
                for x, y in test_loader:
                    # convert data to a vector
                    x = x.view(-1, 784).float()

                    # predict output
                    y_hat = self.predict(x, self.linear_layers)

                    # check correctness
                    _, pred_label = torch.max(y_hat.data, 1)  # torch.max results the max value and the index of the max
                    total_count += x.data.size()[0]  # add the amount of the batch size
                    correct += (pred_label == y.data).sum()  # for each

            accuracy = correct / total_count * 100
            self.accuracy.append(accuracy.item())

        return self.accuracy
