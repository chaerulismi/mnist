import torch
import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class Trainer(object):

  def __init__(self,
               train_dataloader,
               test_dataloader,
               model,
               loss_fn,
               optimizer,
               epochs=5):
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.epochs = epochs
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {self.device} device!')

  def train(self, eval=False):
    self.model.train()
    for epoch in range(1, self.epochs + 1):
      with tqdm(total=len(self.train_dataloader),
                desc=f'Epoch {epoch}') as pbar:
        for X, y in self.train_dataloader:
          X, y = X.to(self.device), y.to(self.device)

          # compute prediction error
          pred = self.model(X)
          loss = self.loss_fn(pred, y)

          # backpropagation
          loss.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()

          pbar.set_postfix({'loss': loss.item()})
          pbar.update()
        if eval:
          self.evaluate()

  def evaluate(self):
    size = len(self.test_dataloader.dataset)
    num_batches = len(self.test_dataloader)
    self.model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
      for X, y in self.test_dataloader:
        X, y = X.to(self.device), y.to(self.device)
        pred = self.model(X)
        test_loss += self.loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
    )
