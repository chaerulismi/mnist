import struct
from array import array

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, images_filepath, labels_filepath, transform=None):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
        self.transform = transform
        self.images, self.labels = self._read_images_labels(self.images_filepath, self.labels_filepath)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def _read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError("Magic number mismatch, expected 2049, got {}".format(magic))
            labels = array('B', f.read())

        with open(images_filepath, 'rb') as f:
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch, expected 2051, got {}".format(magic))
            image_data = array('B', f.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols: (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.array(images, dtype=np.float32), np.array(labels)


if __name__ == '__main__':
    train_images_path = '/Users/bytedance/PycharmProjects/mnist/data/train-images-idx3-ubyte'
    train_labels_path = '/Users/bytedance/PycharmProjects/mnist/data/train-labels-idx1-ubyte'
    transform = lambda x: x / 255.0
    training_data = MNISTDataset(images_filepath=train_images_path, labels_filepath=train_labels_path,
                                 transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    