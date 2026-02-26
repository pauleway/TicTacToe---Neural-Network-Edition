from game import TicTacToeGame
import random 
from copy import deepcopy 

class Player:
  def __init__(self, mark ):
    self.mark = mark

  def get_move(self, game):
    while True:
      try: 
        row, col = input().split(',')
        row = int(row.strip())
        col = int(col.strip())
        if game.position_is_valid(row, col):
          return row,col
        else:
          print("You can't go there - try again!")
      except: 
        print("Invalid input, try again")


class AIRandomPlayer:
  def __init__(self, mark ):
    self.mark = mark

  def get_move(self, game):
    # gather available spaces
    available_spaces = game.get_available_spaces()
    # randomly choose one!
    row, col = random.choice(available_spaces)
    # return row, col
    return (row, col)

class PeekAheadAIPlayer(Player):
    def __init__(self, mark):
        super().__init__(mark)

    def get_move(self, game):
      # gather available spaces 
      available_spaces = game.get_available_spaces()
      # check whether we can win 
      for row, col in available_spaces:
        future_game = TicTacToeGame()
        future_game.board = deepcopy(game.board)
        
        future_game.board[row][col] = self.mark
        if future_game.check_win(self.mark):
          return (row, col)

      # Swap mark
      opposite_mark = 'X' if self.mark == 'O' else 'O'
      # check if opponent can win 
      for row, col in available_spaces:
        future_game = TicTacToeGame()
        future_game.board = deepcopy(game.board)
        future_game.board[row][col] = opposite_mark
        if future_game.check_win(opposite_mark):
          return (row, col)
      
      row, col = random.choice(available_spaces)
      return (row, col)
      
      