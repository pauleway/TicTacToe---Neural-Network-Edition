class TicTacToeGame:
  def __init__(self):
    self.board = [
      ['X','O','X'],
      ['O','X','X'],
      ['O','O','O']
    ]
    self.reset_board()


  def get_available_spaces(self):
    available_spaces = []
    for row in range(3):
      for col in range(3):
        if self.board[row][col] == "_":
          available_spaces.append((row, col))
    return available_spaces

  
  def mark_board(self, mark, row, col):
    self.board[row][col] = mark

    
  def reset_board(self):
      # set each square back to '_'
    for row in range(3):
      for col in range(3):
        self.board[row][col] = "_"

  def print_board(self):
    for row in range(3):
      for col in range(3):
        print(self.board[row][col], end=" ")
      print()
    print()


  # check if position is valid
  def position_is_valid(self, row, col):
      #  check if in bounds
      if row not in range(0,3) or col not in range(0,3):
          return False
      #  check if not already marked
      if self.board[row][col] != '_':
          return False
      return True

  def test_positions(self):
    positions_to_check = [(3,1), (-4,1), (1,5), (1,1), (0,1), (2,2)]
    for row, col in positions_to_check:
      str_command = f"position_is_valid({row},{col})"
      print(str_command + " -> "+str(eval(str_command)))

  def check_win(self, mark):
    VICTORY_PATHS = [
      # Horizontal      
      [(0, 0), (0, 1), (0, 2)],
      [(1, 0), (1, 1), (1, 2)],
      [(2, 0), (2, 1), (2, 2)],
      # Vertical
      [(0, 0), (1, 0), (2, 0)],
      [(0, 1), (1, 1), (2, 1)],
      [(0, 2), (1, 2), (2, 2)],
      # Diagonal
      [(0, 0), (1, 1), (2, 2)],
      [(2, 0), (1, 1), (0, 2)]
      # Input the other possible winning paths
    ]
    for path in VICTORY_PATHS:
      test_marks = []
      for row, col in path:
        # collect all the marks along the path
        test_marks.append(self.board[row][col])
      # count the number of the current 
      # player's mark (X's or O's)
      # if there are three marks along the path, return true
      if test_marks.count(mark) == 3:
        return True
    return False

  def check_tie(self):
    if self.get_available_spaces() == []:
      return True
    else: 
      return False 


