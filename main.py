import argparse
from dataset import MNISTDataset
from model import MLP
from torch import nn
import torch
from trainer import Trainer
from torch.utils.data import DataLoader


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--epochs', type=int, default=1)
  args, _ = parser.parse_known_args()
  return args


def main(args):
  train_images_path = 'data/train-images.idx3-ubyte'
  train_labels_path = 'data/train-labels.idx1-ubyte'
  test_images_path = 'data/t10k-images.idx3-ubyte'
  test_labels_path = 'data/t10k-labels.idx1-ubyte'

  transform = lambda x: x / 255.0

  training_data = MNISTDataset(images_filepath=train_images_path,
                               labels_filepath=train_labels_path,
                               transform=transform)
  test_data = MNISTDataset(images_filepath=test_images_path,
                           labels_filepath=test_labels_path,
                           transform=transform)

  train_dataloader = DataLoader(training_data,
                                batch_size=args.batch_size,
                                shuffle=True)
  test_dataloader = DataLoader(test_data,
                               batch_size=args.batch_size,
                               shuffle=False)

  model = MLP()
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  trainer = Trainer(train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=args.epochs)
  trainer.train(eval=True)


if __name__ == '__main__':
  args = get_args()
  main(args)
