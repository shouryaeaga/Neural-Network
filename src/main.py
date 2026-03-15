from MnistDataLoader import MnistDataLoader
from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

mlp= NeuralNetwork(input_size=784, output_size=10, hidden_layer_size=16, num_hidden_layers=2, softmax=True)
mlp.create_architecture()

weight_layer_1 = np.loadtxt("../pretrained_pt_weights/net_1_weight.csv", delimiter=",")
bias_layer_1 = np.loadtxt("../pretrained_pt_weights/net_1_bias.csv", delimiter=",")
weight_layer_3 = np.loadtxt("../pretrained_pt_weights/net_3_weight.csv", delimiter=",")
bias_layer_3 = np.loadtxt("../pretrained_pt_weights/net_3_bias.csv", delimiter=",")
weight_layer_5 = np.loadtxt("../pretrained_pt_weights/net_5_weight.csv", delimiter=",")
bias_layer_5 = np.loadtxt("../pretrained_pt_weights/net_5_bias.csv", delimiter=",")
mlp.set_layer_weights(weight_layer_1, 0)
mlp.set_layer_weights(weight_layer_3, 2)
mlp.set_layer_weights(weight_layer_5, 4)
mlp.set_layer_bias(bias_layer_1, 0)
mlp.set_layer_bias(bias_layer_3, 2)
mlp.set_layer_bias(bias_layer_5, 4)

mnist = MnistDataLoader("../MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte",
                        "MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

image = X_train[1000]
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
image = np.ndarray.flatten(image)
print(f"Predicted {np.argmax(mlp.forward(image))}")

evaluated = mlp.evaluate(X_train)
correct = np.sum(evaluated == y_train) / len(y_train)
print(f"Correct: {correct}")
wrong_indices = np.where(evaluated != y_train)[0]
image = X_train[wrong_indices[0]]
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
image = np.ndarray.flatten(image)
print(f"Predicted {np.argmax(mlp.forward(image))}")