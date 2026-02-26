"""
Encode the board for the neural network (same as V 0.7).
From the current player's perspective: 1 = my mark, -1 = opponent, 0 = empty.
Move index = row * 3 + col (0-8).
"""
import numpy as np


def board_to_vector(board, my_mark):
  """Flatten 3x3 board to a 9-dim vector from current player's view."""
  vec = []
  other = 'O' if my_mark == 'X' else 'X'
  for row in range(3):
    for col in range(3):
      cell = board[row][col]
      if cell == my_mark:
        vec.append(1.0)
      elif cell == other:
        vec.append(-1.0)
      else:
        vec.append(0.0)
  return np.array(vec, dtype=np.float32)


def move_to_index(row, col):
  return row * 3 + col


def index_to_move(index):
  row = index // 3
  col = index % 3
  return row, col
