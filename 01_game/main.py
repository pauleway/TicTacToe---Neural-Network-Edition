from game import TicTacToeGame
from player import Player, AIRandomPlayer, PeekAheadAIPlayer

def main():
  game = TicTacToeGame()
  players = [
    Player("X"),
    PeekAheadAIPlayer("O")
  ] 

  print("TIC-TAC-TOE!")
  while True:
    for player in players:
      print("input next move in row, col (Ex: 1,0)")
      row, col = player.get_move(game)

      game.mark_board(player.mark, row, col)
      game.print_board()
      if game.check_win(player.mark):
        print(f"Player {player.mark} wins!!!")
        print("Lets Play Again!") 
        game.reset_board()
        break
      if game.check_tie():
        print("Cats game!  No one wins!!!")
        print("Lets Play Again!") 
        game.reset_board()
        break
    

# Run the game. . .
main()