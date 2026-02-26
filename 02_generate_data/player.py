# Copy of V 0.6 players so this folder is self-contained.
from game import TicTacToeGame
import random
from copy import deepcopy


class Player:
  def __init__(self, mark):
    self.mark = mark

  def get_move(self, game):
    while True:
      try:
        row, col = input().split(',')
        row = int(row.strip())
        col = int(col.strip())
        if game.position_is_valid(row, col):
          return row, col
        else:
          print("You can't go there - try again!")
      except Exception:
        print("Invalid input, try again")


class AIRandomPlayer:
  def __init__(self, mark):
    self.mark = mark

  def get_move(self, game):
    available_spaces = game.get_available_spaces()
    return random.choice(available_spaces)


class PeekAheadAIPlayer(Player):
  def __init__(self, mark):
    super().__init__(mark)

  def get_move(self, game):
    available_spaces = game.get_available_spaces()
    for row, col in available_spaces:
      future_game = TicTacToeGame()
      future_game.board = deepcopy(game.board)
      future_game.board[row][col] = self.mark
      if future_game.check_win(self.mark):
        return (row, col)
    opposite_mark = 'X' if self.mark == 'O' else 'O'
    for row, col in available_spaces:
      future_game = TicTacToeGame()
      future_game.board = deepcopy(game.board)
      future_game.board[row][col] = opposite_mark
      if future_game.check_win(opposite_mark):
        return (row, col)
    return random.choice(available_spaces)
