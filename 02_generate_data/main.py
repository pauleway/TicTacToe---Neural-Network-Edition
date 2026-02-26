"""
Step 1: Generate training data for the neural network.

This script plays many games using your Tic-Tac-Toe game and the
"expert" (PeekAhead) player. Every time the expert moves, we save
the board state and the move they chose. That gives us data we'll
use in V 0.8 to train a neural network.

Run this, then go to V 0.8 to train the model and play against it.
"""
from data_generation import generate_data
import numpy as np
import csv


def main():
  print("V 0.7 — Generating training data")
  print("Playing 5,000 games (expert O vs random X)...")
  X, y = generate_data(num_games=5000, seed=42)
  
  with open('training_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['board_0', 'board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'board_6', 'board_7', 'board_8', 'move'])
    for board, move in zip(X, y):
      writer.writerow(list(board) + [int(move)])
  
  print(f"Saved {len(X)} board–move pairs to training_data.csv")
  print()
  print("Next step: Go to V 0.8 and train the model on this file.")


if __name__ == '__main__':
  main()
