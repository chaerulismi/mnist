import torch
import torch.cuda
from torch import nn
from torch.utils.data import DataLoader

from dataset import MNISTDataset
from model import MLP

# Parameters
BATCH_SIZE = 32
EPOCHS = 10
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


if __name__ == '__main__':
    train_images_path = 'data/train-images-idx3-ubyte'
    train_labels_path = 'data/train-labels-idx1-ubyte'
    test_images_path = 'data/t10k-images-idx3-ubyte'
    test_labels_path = 'data/t10k-labels-idx1-ubyte'

    transform = lambda x: x / 255.0

    training_data = MNISTDataset(images_filepath=train_images_path, labels_filepath=train_labels_path,
                                 transform=transform)
    test_data = MNISTDataset(images_filepath=test_images_path, labels_filepath=test_labels_path, transform=transform)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = next(iter(train_dataloader))

    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n--------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
