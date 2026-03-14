import numpy as np
class Linear:
    def __init__(self, input_size, output_size: int):
        self.activations: np.ndarray = np.random.rand(input_size)
        self.W: np.ndarray = np.random.rand(output_size, input_size)
        self.b: np.ndarray = np.random.rand(output_size)
        self.activation_previous = None

    def forward(self, activation_previous: np.ndarray) -> np.ndarray:
        self.activation_previous = activation_previous
        self.activations = np.dot(self.W, activation_previous) + self.b
        return self.activations

# class Weights:
#     def __init__(self, layer_left_size: int, layer_right_size: int):
#         self.layer_left_size = layer_left_size
#         self.layer_right_size = layer_right_size
#         # first row is for the first node on the right, and its weights for all the neurones on the left
#         self.weights: np.ndarray = np.random.rand(layer_right_size, layer_left_size)

class NeuralNetwork:
    def __init__(self, hidden_layer_size: int, output_size: int, input_size: int, num_hidden_layers: int):
        self.weights: list[Weights] = []
        self.layers: list[Linear] = []
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be greater than or equal to 1")
        self.hidden_layer_size: int = hidden_layer_size
        self.output_size: int = output_size
        self.input_size: int = input_size
        self.num_hidden_layers: int = num_hidden_layers

    def create_architecture(self):
        self.layers = [Linear(self.hidden_layer_size)]
        self.weights = [Weights(self.input_size, self.hidden_layer_size)]
        for _ in range(self.num_hidden_layers-1):
            self.layers.append(Linear(self.hidden_layer_size))
            self.weights.append(Weights(self.hidden_layer_size, self.hidden_layer_size))
        self.layers.append(Linear(self.output_size))
        self.weights.append(Weights(self.hidden_layer_size, self.output_size))

    def print_activations(self):
        for layer in self.layers:
            print(layer.neurones)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer, weight in zip(self.layers, self.weights):
            pass
