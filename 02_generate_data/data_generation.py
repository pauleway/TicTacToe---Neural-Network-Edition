"""
Generate training data by playing many games with the "expert" (PeekAhead) player.
We record every board state when it's the expert's turn, and the move they chose.
This gives us (input_board, output_move) pairs to train the neural network.
"""
import numpy as np
from game import TicTacToeGame
from player import PeekAheadAIPlayer, AIRandomPlayer
from board_utils import board_to_vector, move_to_index


def play_one_game(game, expert_mark, expert_player, other_player, records):
  """Play one game; append (board_vec, move_index) for each expert move."""
  while True:
    for mark, player in [('X', expert_player if expert_mark == 'X' else other_player),
                         ('O', expert_player if expert_mark == 'O' else other_player)]:
      if game.check_win('X') or game.check_win('O') or game.check_tie():
        return
      row, col = player.get_move(game)
      if player is expert_player:
        board_vec = board_to_vector(game.board, expert_mark)
        move_idx = move_to_index(row, col)
        records['boards'].append(board_vec)
        records['moves'].append(move_idx)
      game.mark_board(mark, row, col)
      if game.check_win(mark) or game.check_tie():
        return


def generate_data(num_games=10_000, expert_mark='O', use_two_experts=False, seed=42):
  """
  Run many games and collect expert moves.
  expert_mark: which side the expert plays ('X' or 'O').
  use_two_experts: if True, both sides are expert (more strong data).
  """
  np.random.seed(seed)
  records = {'boards': [], 'moves': []}
  expert = PeekAheadAIPlayer(expert_mark)
  other = PeekAheadAIPlayer('X' if expert_mark == 'O' else 'O') if use_two_experts else AIRandomPlayer('X' if expert_mark == 'O' else 'O')

  for _ in range(num_games):
    game = TicTacToeGame()
    play_one_game(game, expert_mark, expert, other, records)

  X = np.stack(records['boards'])
  y = np.array(records['moves'], dtype=np.int64)
  return X, y


def main():
  import argparse
  import csv
  p = argparse.ArgumentParser(description='Generate Tic-Tac-Toe training data')
  p.add_argument('--games', type=int, default=10_000, help='Number of games to play')
  p.add_argument('--expert', choices=['X', 'O'], default='O', help='Which side is the expert')
  p.add_argument('--both-experts', action='store_true', help='Use expert on both sides')
  p.add_argument('--out', default='training_data.csv', help='Output file')
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()

  print(f'Playing {args.games} games (expert={args.expert}, both_experts={args.both_experts})...')
  X, y = generate_data(num_games=args.games, expert_mark=args.expert, use_two_experts=args.both_experts, seed=args.seed)
  
  with open(args.out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['board_0', 'board_1', 'board_2', 'board_3', 'board_4', 'board_5', 'board_6', 'board_7', 'board_8', 'move'])
    for board, move in zip(X, y):
      writer.writerow(list(board) + [int(move)])
  
  print(f'Saved {len(X)} samples to {args.out}')


if __name__ == '__main__':
  main()
