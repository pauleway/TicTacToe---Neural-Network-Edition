# Step 2 — Generating Training Data

## Learning goals

- Understand why game state must be converted to numbers before a neural network can use it
- See how "imitation learning" works: record what an expert does, use that as training data
- Understand the structure of a training dataset: inputs (X) and labels (y)

---

## What we do

This folder is **only about creating the dataset** we need to train a neural network later. No training happens here.

1. Use the Tic-Tac-Toe game and the **PeekAhead** "expert" player from Step 1.
2. Play many games (expert O vs random X).
3. Every time the expert is about to move, record:
   - **Board state** (encoded as 9 numbers: 1 = my mark, -1 = opponent, 0 = empty)
   - **The move they chose** (as a number 0–8 for the cell)
4. Save all those (board, move) pairs to `training_data.csv`.

That file is the **training data** for Step 3.

**After running this step, open `training_data.csv` in Excel, Numbers, or Google Sheets.**
You'll see ~22,000 rows. Each row is one board state + the expert's move. This is what
the neural network will learn from.

---

## How to run

```bash
python main.py
```

Plays 5,000 games and creates `training_data.csv` with a header row.

**Expected output:**
```
Playing 5,000 games (expert O vs random X)...
Saved 22148 board–move pairs to training_data.csv

Next step: Go to 03_train_and_play and train the model on this file.
```

(Your exact number of samples may vary slightly.)

To customize (e.g. more games), edit the variables at the top of `data_generation.py` and run `python data_generation.py`.

---

## Tutorial: How the code works

This section walks through each part of the code so you can see exactly how we build the dataset.

### 1. Why we need numbers: `board_utils.py`

The neural network in Step 3 can't work with a 3×3 grid of characters like `'X'`, `'O'`, `'_'`. It needs a **fixed-length list of numbers**. So we define a standard encoding:

- **From the current player's point of view:**
  - `1.0` = my mark
  - `-1.0` = opponent's mark
  - `0.0` = empty

We also need to turn a move "cell (row, col)" into a single number 0–8 so the network can predict "which of 9 cells?" We use:

- **Cell index** = `row * 3 + col`
  So (0,0)→0, (0,1)→1, …, (2,2)→8.

**`board_to_vector(board, my_mark)`**
We loop over the 3×3 board and build a list of 9 floats. For each cell we check: is it `my_mark`? Then 1.0. Is it the other player? Then -1.0. Otherwise 0.0. We return that as a NumPy array. The network will always see the board from "the player whose turn it is," so one model can learn to play for either side if we train it that way.

**`move_to_index(row, col)`**
Returns `row * 3 + col` (0–8).

**`index_to_move(idx)`**
The reverse: `row = idx // 3`, `col = idx % 3`. Used in Step 3 when the network outputs an index and we need (row, col) for the game.

---

### 2. Playing one game and recording: `data_generation.py` — `play_one_game`

We need to run many games and, **only when it's the expert's turn**, save (board, move) before the move is played.

```python
def play_one_game(game, expert_player, other_player, records):
```

- **`game`** – A fresh `TicTacToeGame()` (empty board).
- **`expert_player`** – The PeekAhead player (always plays O).
- **`other_player`** – The other player (random or another expert).
- **`records`** – A dict with lists `'boards'` and `'moves'` we append to.

The loop alternates between X and O. For each turn:

1. **Check if the game is already over** (win or tie). If so, return.
2. **Ask the current player for a move** with `player.get_move(game)` → `(row, col)`.
3. **If the current player is the expert**, we record:
   - **Board** – Encode the board *before* the move using `board_to_vector(game.board, 'O')`. That's the "state" the expert saw.
   - **Move** – Convert (row, col) to an index with `move_to_index(row, col)`.
   - Append both to `records['boards']` and `records['moves']`.
4. **Apply the move** with `game.mark_board(mark, row, col)`.
5. **Check again** if the game is over after this move; if so, return.

So every expert move produces one (board vector, move index) pair. We never record the random player's moves—only the expert's, because we want to learn "what would a good player do in this situation?"

---

### 3. Running many games: `data_generation.py` — `generate_data`

```python
def generate_data(num_games=10_000, seed=42):
```

- **`num_games`** – How many full games to play.
- **`seed`** – Random seed so runs are reproducible.

We set the random seed, create an **expert** (PeekAhead playing O) and a **random** opponent (X). Then we run `num_games` games. For each game we create a new `TicTacToeGame()`, call `play_one_game(...)`, and the `records` dict gets filled with all expert (board, move) pairs across all games.

At the end we convert the lists to arrays:

- **`X`** = `np.stack(records['boards'])` → shape `(N, 9)` (N = total number of expert moves).
- **`y`** = `np.array(records['moves'], dtype=np.int64)` → shape `(N,)` (each value 0–8).

We return `X` and `y`. The script's `main()` then saves them to `training_data.csv` with a header row (`board_0` through `board_8`, then `move`) so you can inspect them in any spreadsheet app.

---

### 4. Entry point: `main.py`

`main.py` is the simplest path:

1. Call `generate_data(num_games=5000, seed=42)`.
2. Save the result to `training_data.csv` with a header row.
3. Print how many samples were saved and remind you to go to Step 3 next.

So "run `main.py`" = "generate 5,000 games and create `training_data.csv` in one step."

---

## Files in this folder

| File | Purpose |
|------|---------|
| `game.py`, `player.py` | Same as Step 1 (the game and PeekAhead / Random players). |
| `board_utils.py` | Encodes the board as 9 numbers; converts (row, col) ↔ index 0–8. |
| `data_generation.py` | Runs the games and records (board, move); can be run from the command line. |
| `main.py` | Simple entry point: runs 5,000 games and saves `training_data.csv`. |

---

## What to try

These are optional experiments — great for exploration or extra credit:

- **Fewer games:** Edit `num_games = 500` in `data_generation.py`, run it, then use that smaller dataset in Step 3. Does the network play differently?
- **More games:** Try `num_games = 20000`. Does the trained network improve meaningfully?
- **Look at the CSV:** Open `training_data.csv` in a spreadsheet. Find a row where `move = 4` (the center cell). What does the board look like in that row? Does the expert often play the center?

---

## Next step

Copy `training_data.csv` to the `03_train_and_play/` folder, then go there to train the model.
