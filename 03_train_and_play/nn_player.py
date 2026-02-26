"""
Player that uses the trained neural network to choose moves.
Loads the model, encodes the board, runs forward pass, and picks the best legal cell.
"""
import torch
import numpy as np
from train_model import TicTacToeNet
from board_utils import board_to_vector, index_to_move


class NNPlayer:
  def __init__(self, mark, model_path='model.pt', device=None):
    self.mark = mark
    if device is None:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.device = device
    self.model = TicTacToeNet()
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
      state = state['state_dict']
    self.model.load_state_dict(state)
    self.model.to(device)
    self.model.eval()

  def get_move(self, game):
    vec = board_to_vector(game.board, self.mark)
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
    with torch.no_grad():
      logits = self.model(x).squeeze(0).cpu().numpy()
    available = game.get_available_spaces()
    for row in range(3):
      for col in range(3):
        if (row, col) not in available:
          logits[row * 3 + col] = -np.inf
    idx = int(np.argmax(logits))
    return index_to_move(idx)
