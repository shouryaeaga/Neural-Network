# MLP Visualiser

A multi-layer perceptron built from scratch in Python and NumPy, trained on MNIST handwritten digits. An interactive Tkinter GUI lets you step through a forward pass, inspect neuron activations per layer, and view weight heatmaps showing what each neuron responds to.

Built as an A-level Computer Science NEA project.

---

## Features

- Forward pass implemented in pure NumPy
- Compatible with pre-trained weights exported from equivalent PyTorch architectures
- Step-through forward pass: watch activations propagate layer by layer
- Per-layer neuron activation display, updated live during inference
- Weight heatmaps showing the learned receptive field of each neuron
- Output probability bar chart across all 10 digit classes
- Interactive node inspection: click any neuron to see its weights and activation value
- Draw-your-own digit canvas for live inference (stretch goal)

---

## Architecture

784 inputs (28x28 flattened pixel values) → 16 → 16 → 10 outputs (one per digit class)

- Activation: ReLU
- Loss: Mean Squared Error (MSE)
- Optimiser: SGD with mini-batches
- Data: MNIST loaded from binary IDX format

---

## TODO

### Core Network

- [x] Forward pass
- [x] ReLU activation
- [x] He weight initialisation
- [x] Load and apply pre-trained weights from equivalent PyTorch model
- [x] Verified correct predictions against PyTorch reference on MNIST test samples
- [ ] MSE loss function
- [ ] Backpropagation
- [ ] Gradient descent weight update

### Training

- [ ] SGD training loop
- [ ] Mini-batch SGD
- [ ] Training/validation split
- [ ] Accuracy and loss logging per epoch
- [ ] Model save and load
- [ ] MNIST binary IDX data loader and preprocessor

### GUI / Visualiser

- [ ] Main Tkinter application window
- [ ] Step-through forward pass with per-layer activation display
- [ ] Live neuron activation colouring
- [ ] Weight heatmap view (28x28 grid per neuron in hidden layer 1)
- [ ] Output probability bar chart
- [ ] Interactive node inspection on click

### Extras / Extensions

- [ ] Draw-your-own digit canvas with live inference
- [ ] Configurable network depth and width at runtime
- [ ] Training curve plot (loss vs. epoch)
