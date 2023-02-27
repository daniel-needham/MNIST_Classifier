import nn_classifier
import numpy as np
import matplotlib as plt

epoch = 20
max_layers = 5
results = []
layers = [784, 10]
hidden_nodes = 32

for x in range(max_layers):
    nn = nn_classifier.nnClassifier(layers, epoch)
    results.append(nn.train_and_test())

    layers.insert(1, hidden_nodes)

print(results)

file = open('result.txt', 'w')
for item in results:
    file.write(str(item) + '\n')
file.close()
