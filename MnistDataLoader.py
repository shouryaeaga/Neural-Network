from array import array
import numpy as np
import struct

# taken from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
class MnistDataLoader:
    def __init__(self, train_path_images: str, train_path_labels: str, test_path_images: str, test_path_labels: str):
        self.train_path_images = train_path_images
        self.train_path_labels = train_path_labels
        self.test_path_images = test_path_images
        self.test_path_labels = test_path_labels

    def read_images_labels(self, label_path: str, img_path: str):
        labels = []
        with open(label_path, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise Exception("Not a Mnist file.")
            labels = array("B", f.read())

        with open(img_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise Exception("Not a Mnist file.")
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            img = np.divide(img, 255)
            images[i][:] = img
        images = np.array(images)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_path_labels, self.train_path_images)
        x_test, y_test = self.read_images_labels(self.test_path_labels, self.test_path_images)
        return (x_train, y_train),(x_test, y_test)