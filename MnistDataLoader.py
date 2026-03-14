import numpy as np


class MnistDataLoader:
    def __init__(self, train_path_images: str, train_path_labels: str, test_path_images: str, test_path_labels: str):
        self.train_path_images = train_path_images
        self.train_path_labels = train_path_labels
        self.test_path_images = test_path_images
        self.test_path_labels = test_path_labels

    def load_training(self):
        with open(self.train_path_images, 'rb') as f:
            data = f.read()
        magic = int.from_bytes(data[:4], byteorder='little')
        print(magic)
