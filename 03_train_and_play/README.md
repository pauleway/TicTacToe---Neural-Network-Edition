# Step 3 — Training the Model and Playing

## Learning goals

- See a real neural network training loop in practice (forward pass, loss, backprop, optimizer step)
- Understand the difference between a learned AI (the network) and a rule-based AI (PeekAhead)
- Trace the full pipeline: board → numbers → network → move

---

## What we do

This folder is about **training a neural network** on the data from Step 2, then **playing against it**.

You need the file **`training_data.csv`** from Step 2. Copy it into this folder first.

1. **Train** – Load the (board, move) pairs from `training_data.csv`. Train a small network to predict the expert's move from the board. Save the trained model as `model.pt`.
2. **Play** – Use `model.pt` as a player: given the current board, the network picks a cell (we only allow legal moves). You play as X, the NN plays as O.

---

## How to run

### Step 1: Train the model

```bash
python train_model.py --data training_data.csv --epochs 30 --save model.pt
```

- **Input**: 9 numbers (board from current player's view).
- **Output**: 9 "scores" (one per cell); we pick the highest among legal cells.
- **Architecture**: 9 → 64 → 64 → 9, ReLU, cross-entropy loss.

**Expected output:**
```
Epoch 1/30  Loss: 1.8432
Epoch 5/30  Loss: 1.4217
Epoch 10/30  Loss: 1.2803
Epoch 15/30  Loss: 1.1944
Epoch 20/30  Loss: 1.1421
Epoch 25/30  Loss: 1.1082
Epoch 30/30  Loss: 1.0891
Saved model to model.pt
```

You'll see the loss decrease over epochs — that means the network is learning to predict the expert's moves.

### Step 2: Play against the NN

```bash
python main.py
```

You are X; the NN is O. Enter moves as `row,col` (e.g. `1,0`).

---

## Tutorial: How the code works

This section walks through each part of the code so you can see how we train the network and use it to play.

### 1. The network: `train_model.py` — `TicTacToeNet`

We need a function that takes 9 numbers (the board vector) and outputs 9 numbers (one per cell). The cell with the **highest** output will be our chosen move (after we mask illegal cells in the player).

```python
class TicTacToeNet(nn.Module):
  def __init__(self, hidden=64):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(9, hidden),   # 9 inputs -> 64 units
      nn.ReLU(),
      nn.Linear(hidden, hidden),  # 64 -> 64
      nn.ReLU(),
      nn.Linear(hidden, 9),    # 64 -> 9 outputs
    )

  def forward(self, x):
    return self.net(x)
```

- **`nn.Linear(9, hidden)`** – First layer: 9 inputs (the board), 64 outputs. Each output is a weighted sum of the 9 inputs plus a bias.
- **`nn.ReLU()`** – ReLU(x) = max(0, x). Adds non-linearity so the network can learn more than a simple linear map.
- The second **`nn.Linear(hidden, hidden)`** and **ReLU** add another hidden layer (64→64).
- **`nn.Linear(hidden, 9)`** – Final layer: 64 → 9. These 9 numbers are the **logits** (raw scores) for each cell. We don't apply softmax in the forward pass because PyTorch's `CrossEntropyLoss` does that internally.

So: **input** = one board vector of length 9, **output** = 9 logits. The index of the maximum logit is the "predicted move" (we'll enforce legality when playing).

---

### 2. Loading data and training loop: `train_model.py` — `train`

**Load the dataset**

```python
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        X_list.append([float(x) for x in row[:9]])
        y_list.append(int(row[9]))

X = torch.tensor(X_list, dtype=torch.float32)   # shape (N, 9)
y = torch.tensor(y_list, dtype=torch.long)       # shape (N,) values 0-8
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

- `X` is the board vectors from Step 2; `y` is the move index (0–8) the expert chose for each board.
- `TensorDataset` pairs each row of X with the corresponding label in y.
- `DataLoader` gives us **batches** of (X_batch, y_batch). Shuffling helps the model see a mix of situations each epoch.

**Model, loss, optimizer**

- **Model** – `TicTacToeNet()` moved to GPU if available (`model.to(device)`).
- **Loss** – `nn.CrossEntropyLoss()`. For each sample it compares the 9 logits to the true class (0–8): it applies log-softmax and returns the negative log probability of the correct class. We minimize that.
- **Optimizer** – `torch.optim.Adam(model.parameters(), lr=lr)`. Adam updates the weights to reduce the loss.

**One epoch**

For each batch:

1. Move `batch_X` and `batch_y` to the device (CPU or GPU).
2. **`optimizer.zero_grad()`** – Clear old gradients (PyTorch accumulates them by default).
3. **`logits = model(batch_X)`** – Forward pass: (batch_size, 9) logits.
4. **`loss = criterion(logits, batch_y)`** – Scalar loss for this batch.
5. **`loss.backward()`** – Compute gradients of the loss with respect to every weight.
6. **`optimizer.step()`** – Update the weights using those gradients.

After all batches, we've seen the whole dataset once (one epoch). We repeat for `epochs` times and periodically print the average loss. When training is done we save the weights with `torch.save({'state_dict': model.state_dict()}, save_path)` so we can load them later without redefining the architecture.

---

### 3. Using the trained model to play: `nn_player.py` — `NNPlayer`

**Initialization (`__init__`)**

- We store `mark` ('X' or 'O') and choose CPU or GPU.
- We create a **new** `TicTacToeNet()` (same architecture as in `train_model.py`), then load the saved weights from `model.pt`. The file may store either `{'state_dict': ...}` or just the state dict; we handle both.
- **`model.load_state_dict(state)`** – Copies the trained weights into the network.
- **`model.eval()`** – Puts the model in evaluation mode (turns off dropout if we had any; for this small net it mainly signals "we're not training").

**Getting a move (`get_move(game)`)**

1. **Encode the board** – `board_to_vector(game.board, self.mark)` gives a 9-element vector (same encoding as Step 2). The NN always sees the board from "our" perspective (1 = us, -1 = opponent, 0 = empty).
2. **Run the network** – Convert the vector to a tensor, add a batch dimension with `unsqueeze(0)` (shape 1×9), move to device, then `model(x)`. We use **`torch.no_grad()`** so we don't compute gradients (saves memory and time). The output is shape (1, 9); we squeeze to (9,) and convert to NumPy: `logits`.
3. **Mask illegal moves** – The network can assign a high score to an occupied cell. We get `game.get_available_spaces()` and set `logits[row*3+col] = -np.inf` for every (row, col) that is **not** in that list. So only empty cells can win.
4. **Choose the best cell** – `idx = np.argmax(logits)` (0–8). We convert back to (row, col) with `index_to_move(idx)` and return that as the move.

So each time the NN has to move, we: encode board → forward pass → mask illegal → pick argmax → return (row, col). The game loop in `main.py` then calls `game.mark_board(player.mark, row, col)` just like for any other player.

---

### 4. Same encoding as Step 2: `board_utils.py`

This file is the same as in Step 2. The network was trained on vectors from **`board_to_vector(board, my_mark)`**. When we play, we must use the **exact same encoding** for the same player perspective, or the model would get the wrong kind of input. So we use the same 1 / -1 / 0 convention and the same (row, col) ↔ index mapping.

---

### 5. Game loop: `main.py`

`main.py` creates a `TicTacToeGame()` and two players: a human `Player("X")` and `NNPlayer("O", model_path='model.pt')`. Then it runs the usual loop: for each player in turn, print board, ask for a move (`player.get_move(game)`), mark the board, check win/tie, reset if the game is over. For the human, `get_move` reads from the keyboard; for the NN, `get_move` runs the steps in the previous section and returns (row, col). The game loop is identical to Step 1 — the only change is one of the players is now a neural network.

---

## Files in this folder

| File | Purpose |
|------|---------|
| `game.py`, `player.py` | Same as Steps 1 and 2 (game and human/random players). |
| `board_utils.py` | Same encoding as Step 2 (board → 9 numbers). |
| `train_model.py` | Defines the network, loads CSV data, trains, saves `model.pt`. |
| `nn_player.py` | Loads `model.pt`, encodes board, runs model, picks best legal move. |
| `main.py` | Human vs NN game loop. |

---

## What to try

These are optional experiments — great for exploration or extra credit:

- **Fewer epochs:** Train with `--epochs 5` and play. Does the weaker network make obvious mistakes?
- **More epochs:** Try `--epochs 100`. Does the network improve, or does it plateau?
- **Smaller dataset:** Use a `training_data.csv` generated with only 500 games (from Step 2). How does the network play compared to one trained on 5,000 games?
- **Quick demo:** For a faster run (useful for in-class demos):
  ```bash
  python train_model.py --data training_data.csv --epochs 15 --save model.pt
  python main.py
  ```
  The NN will be a bit weaker but trains in half the time.
