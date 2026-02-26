"""
Step 2: Train the model, then play against it.

First, copy training_data.csv from V 0.7 into this folder (or run V 0.7 first).
Then:
  1. Train:  python train_model.py --data training_data.csv --save model.pt
  2. Play:   python main.py
"""
from game import TicTacToeGame
from player import Player
from nn_player import NNPlayer


def main():
  game = TicTacToeGame()
  players = [
    Player("X"),       # Human
    NNPlayer("O", model_path='model.pt')
  ]

  print("TIC-TAC-TOE vs Neural Network!")
  print("You are X. Enter moves as row,col (e.g. 1,0)")
  print()
  while True:
    for player in players:
      game.print_board()
      print(f"Player {player.mark}'s turn.")
      row, col = player.get_move(game)
      game.mark_board(player.mark, row, col)
      if game.check_win(player.mark):
        game.print_board()
        print(f"Player {player.mark} wins!")
        print("Play again!")
        game.reset_board()
        break
      if game.check_tie():
        game.print_board()
        print("Tie game!")
        print("Play again!")
        game.reset_board()
        break


if __name__ == '__main__':
  main()
