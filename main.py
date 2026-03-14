from MnistDataLoader import MnistDataLoader
from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

mlp= NeuralNetwork(input_size=784, output_size=10, hidden_layer_size=16, num_hidden_layers=2, softmax=True)
mlp.create_architecture()
image = np.random.rand(28, 28)

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

image = np.ndarray.flatten(image)

print(np.argmax(mlp.forward(image)))

mnist = MnistDataLoader("train-images.idx3-ubyte.zip", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte.zip", "t10k-labels.idx1-ubyte")
mnist.load_training()