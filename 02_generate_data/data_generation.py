"""
Generate training data by playing many games with the "expert" (PeekAhead) player.
We record every board state when it's the expert's turn, and the move they chose.
This gives us (input_board, output_move) pairs to train the neural network.
"""
import numpy as np
from game import TicTacToeGame
from player import PeekAheadAIPlayer, AIRandomPlayer
from board_utils import board_to_vector, move_to_index


def play_one_game(game, expert_player, other_player, records):
  while True:
    for mark, player in [('X', other_player), ('O', expert_player)]:
      if game.check_win('X') or game.check_win('O') or game.check_tie():
        return
      row, col = player.__________________________
      if player is expert_player:
        board_vec = board_to_vector(game.board, 'O')
        move_idx = move_to_index(row, col)
        records['boards']._______________________
        records['moves'].append(move_idx)
      game.mark_board(mark, row, col)
      if game.check_win(mark) or game.check_tie():
        return

def generate_data(num_games=10_000, seed=42):
  """
  Run many games and collect expert moves.
  Expert plays O, opponent is random.
  """
  np.random.seed(seed)
  records = {'boards': [], 'moves': []}
  expert = PeekAheadAIPlayer('O')
  other = AIRandomPlayer('X')

  ___________________________________
    game = TicTacToeGame()
    play_one_game(game, expert, other, records)

  X = np.stack(records['boards'])
  y = np.array(records['moves'], dtype=np.int64)
  return X, y

def main():
  import csv
  num_games = 10_000
  output_file = 'training_data.csv'
  seed = 42

  print(f'Playing {num_games} games (expert=O vs random X)...')
  X, y = generate_data(num_games=num_games, seed=seed)
  
  with open(output_file, 'w', newline='') as f:
    writer = ______________________
    writer.writerow(['board_0', 'board_1', 'board_2', 'board_3', 'board_4', 
				'board_5', 'board_6', 'board_7', 'board_8', 'move'])
    for board, move in zip(X, y):
      writer._________________________________________________
  
  print(f'Saved {len(X)} samples to {output_file}')


if __name__ == '__main__':
  main()
