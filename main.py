from NeuralNetwork import NeuralNetwork

mlp= NeuralNetwork(input_size=28, output_size=10, hidden_layer_size=16, num_hidden_layers=2)
mlp.create_architecture()
mlp.print_activations()