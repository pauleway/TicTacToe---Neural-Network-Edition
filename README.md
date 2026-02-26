# Tic-Tac-Toe: Teaching a Neural Network to Play

You already built a Tic-Tac-Toe AI that uses if-statements and lookahead rules to decide its moves.
Now you're going to build one that **learns by watching** — without being told the rules.

You'll record thousands of games played by your expert AI, then train a small neural network to
imitate it. This technique — called **imitation learning** or **behavioral cloning** — is used in
real-world AI systems including robotic manipulation and game-playing agents.

---

## What you'll build

This project has three steps, each in its own folder:

| Step | Folder | What happens |
|------|--------|--------------|
| 1 | `01_game/` | Play the finished game with the expert PeekAhead AI. This is the foundation. |
| 2 | `02_generate_data/` | Run thousands of games and record every move the expert makes. Save to a CSV file. |
| 3 | `03_train_and_play/` | Train a neural network on that CSV. Then play against it. |

---

## Learning objectives

By the end of this project you will be able to:

- Explain why game state needs to be converted to numbers before a neural network can use it
- Describe what "training data" is and where it comes from in this project
- Trace the flow of data from a game board → vector → neural network → move
- Read a basic PyTorch training loop and identify what each part does
- Explain the difference between a rule-based AI (PeekAhead) and a learned AI (the neural network)
- Define "imitation learning" and give a real-world example of where it's used

---

## Prerequisites

- Python 3.8 or higher
- Comfortable with Python classes, methods, and loops (covered in earlier units)
- No prior machine learning experience needed

---

## One-time setup

Do this once before starting Step 1.

### 1. Check your Python version

```bash
python --version
```

You need **3.8 or higher**. If the command isn't found, try `python3 --version`. If you're on
Windows and neither works, install Python from [python.org](https://python.org).

### 2. Create a virtual environment

A virtual environment keeps the libraries for this project separate from everything else on your computer.

```bash
# Navigate to this project folder first, then:
python -m venv venv
```

If `python` doesn't work, replace it with `python3` everywhere below.

### 3. Activate the virtual environment

Every time you open a new terminal to work on this project, you need to activate it first:

```bash
# Mac / Linux
source venv/bin/activate

# Windows — Command Prompt
venv\Scripts\activate

# Windows — PowerShell
.\venv\Scripts\Activate.ps1
```

When it's active you'll see `(venv)` at the start of your terminal prompt.

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

This installs NumPy (for arrays) and PyTorch (for the neural network). It may take a minute.

### 5. Verify setup

```bash
python 01_game/main.py
```

You should see a Tic-Tac-Toe board and be able to play a game. If this works, you're ready.

---

## The three steps

### Step 1 — Play the game (`01_game/`)

```bash
python 01_game/main.py
```

Read `01_game/README.md` before you start. You're playing as X against the PeekAhead AI (O).
The PeekAhead AI will be your "expert" — the teacher for the neural network in the next steps.

**Estimated time: 5–10 minutes**

---

### Step 2 — Generate training data (`02_generate_data/`)

```bash
cd 02_generate_data
python main.py
```

This plays 5,000 games automatically and saves every expert move to `training_data.csv`.
Read `02_generate_data/README.md` to understand what's in that file.

**Tip:** Open `training_data.csv` in Excel, Numbers, or Google Sheets after it runs.
Each row is one board state + the move the expert chose. This is your training data.

**Estimated time: under 1 minute to run**

---

### Step 3 — Train the model and play (`03_train_and_play/`)

Copy `training_data.csv` from `02_generate_data/` into the `03_train_and_play/` folder, then:

```bash
cd 03_train_and_play
python train_model.py --data training_data.csv --epochs 30 --save model.pt
python main.py
```

Read `03_train_and_play/README.md` to understand what the training loop is doing.

**Estimated time: 1–3 minutes to train, then you can play**

---

## Reflection questions

Answer these as part of your submission:

1. The PeekAhead AI uses if-statements to decide its moves. The neural network doesn't contain
   any game rules. What does it "know" instead, and where does that knowledge come from?

2. Open `training_data.csv` in a spreadsheet. What do the values 1.0, -1.0, and 0.0 represent
   in the board columns? Why do we encode it this way instead of using 'X', 'O', and '_'?

3. Try regenerating the training data with only 500 games (see `02_generate_data/README.md`
   for the command). Then retrain and play. Does the network play better or worse? Why?

4. Can the neural network ever beat PeekAhead consistently? Why or why not? Think about what
   the network learned and what PeekAhead "knows."

5. This approach — training an AI by showing it expert examples — is called **imitation learning**
   or **behavioral cloning**. Name one real-world application where you think this technique
   would be useful, and explain why it's a good fit.

---

## Common problems

**`python: command not found`**
Use `python3` instead of `python` throughout.

**`ModuleNotFoundError: No module named 'torch'` or `'numpy'`**
You either forgot to activate your virtual environment (`source venv/bin/activate`) or ran
`pip install` outside of it. Activate venv first, then re-run `pip install -r requirements.txt`.

**`(venv)` is not showing in my prompt**
Your virtual environment isn't active. Run the activate command from step 3 again.

**Training is slow / taking a long time**
With 5,000 games and 30 epochs on a laptop CPU, training typically takes 30–90 seconds total.
If it's taking much longer, that's unusual — check that you don't have a very old machine.

**Windows: `cd 02_generate_data` uses backslashes**
On Windows Command Prompt, use `cd 02_generate_data` (forward slashes work too in most terminals).
If you get a path error, try `cd "02_generate_data"`.
