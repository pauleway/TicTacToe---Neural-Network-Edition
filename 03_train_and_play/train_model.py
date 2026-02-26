"""
Step 2: Train a neural network on the data from V 0.7.

Input: 9 numbers (board from current player's view: 1=me, -1=opponent, 0=empty).
Output: 9 classes (which cell to play, 0-8).

We train the network to predict the expert's move for each board in the dataset.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import csv

class TicTacToeNet(nn.Module):
  """Simple feedforward net: 9 -> 64 -> 64 -> 9."""

  def __init__(self, hidden=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(9, hidden),
      nn.ReLU(),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Linear(hidden, 9),
    )

  def forward(self, x):
    return self.net(x)


def train(data_path='training_data.csv', epochs=30, batch_size=64, lr=0.001, save_path='model.pt', seed=42):
  torch.manual_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  X_list = []
  y_list = []
  with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
      X_list.append([float(x) for x in row[:9]])
      y_list.append(int(row[9]))
  
  X = torch.tensor(X_list, dtype=torch.float32)
  y = torch.tensor(y_list, dtype=torch.long)
  dataset = TensorDataset(X, y)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  model = TicTacToeNet().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  model.train()
  for epoch in range(epochs):
    total_loss = 0.0
    for batch_X, batch_y in loader:
      batch_X, batch_y = batch_X.to(device), batch_y.to(device)
      optimizer.zero_grad()
      logits = model(batch_X)
      loss = criterion(logits, batch_y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    if (epoch + 1) % 5 == 0 or epoch == 0:
      print(f'Epoch {epoch + 1}/{epochs}  Loss: {total_loss / len(loader):.4f}')

  torch.save({'state_dict': model.state_dict()}, save_path)
  print(f'Saved model to {save_path}')
  return model


def main():
  import argparse
  p = argparse.ArgumentParser(description='Train the Tic-Tac-Toe neural network')
  p.add_argument('--data', default='training_data.csv', help='Path to .csv file from V 0.7')
  p.add_argument('--epochs', type=int, default=30)
  p.add_argument('--batch', type=int, default=64)
  p.add_argument('--lr', type=float, default=0.001)
  p.add_argument('--save', default='model.pt')
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()

  train(data_path=args.data, epochs=args.epochs, batch_size=args.batch, lr=args.lr, save_path=args.save, seed=args.seed)


if __name__ == '__main__':
  main()
