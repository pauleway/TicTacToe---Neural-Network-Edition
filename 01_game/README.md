# Step 1 — The Game

## Learning goals

- Understand how the finished Tic-Tac-Toe game is structured (game logic, player classes, game loop)
- Play against the **PeekAhead AI** — the "expert" whose moves you'll collect in Step 2
- Understand *why* the PeekAhead strategy works: it checks if it can win, then if it needs to block, then picks randomly

---

## How to run

```bash
python 01_game/main.py
```

You play as **X**. The PeekAhead AI plays as **O**. Enter moves as `row,col` (e.g. `1,1` for the center).

---

## Files

| File | Purpose |
|------|---------|
| `game.py` | The `TicTacToeGame` class — board, validation, win/tie checking |
| `player.py` | `Player` (human), `AIRandomPlayer`, and `PeekAheadAIPlayer` |
| `main.py` | Sets up the players and runs the game loop |

---

## The PeekAhead strategy

Open `player.py` and read the `PeekAheadAIPlayer.get_move()` method. It does three things in order:

1. **Try to win** — Check every empty cell. If placing there wins the game, play it.
2. **Block the opponent** — Check if the opponent can win on their next move. If so, block that cell.
3. **Pick randomly** — If neither of the above applies, pick any empty cell at random.

This is a simple one-move lookahead, not a full search tree. It's good but not perfect — it
can occasionally be beaten if you set up a "fork" (two ways to win at once).

**Think about it:** Why is this strategy "expert enough" to teach a neural network?
What would happen if we used the random player as the teacher instead?

---

## Next step

Once you've played a few games and understand how PeekAhead works, go to **`02_generate_data/`**.
