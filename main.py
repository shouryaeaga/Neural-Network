from MnistDataLoader import MnistDataLoader
from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

mlp= NeuralNetwork(input_size=784, output_size=10, hidden_layer_size=16, num_hidden_layers=2, softmax=True)
mlp.create_architecture()

weight_layer_1 = np.loadtxt("net_1_weight.csv", delimiter=",")
bias_layer_1 = np.loadtxt("net_1_bias.csv", delimiter=",")
weight_layer_3 = np.loadtxt("net_3_weight.csv", delimiter=",")
bias_layer_3 = np.loadtxt("net_3_bias.csv", delimiter=",")
weight_layer_5 = np.loadtxt("net_5_weight.csv", delimiter=",")
bias_layer_5 = np.loadtxt("net_5_bias.csv", delimiter=",")
mlp.set_layer_weights(weight_layer_1, 0)
mlp.set_layer_weights(weight_layer_3, 2)
mlp.set_layer_weights(weight_layer_5, 4)
mlp.set_layer_bias(bias_layer_1, 0)
mlp.set_layer_bias(bias_layer_3, 2)
mlp.set_layer_bias(bias_layer_5, 4)

mnist = MnistDataLoader("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

image = X_train[1]
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
image = np.ndarray.flatten(image)
print(f"Predicted {np.argmax(mlp.forward(image))}")