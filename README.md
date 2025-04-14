# Chinese Checkers MCTS Implementation

This repository contains an implementation of Monte Carlo Tree Search (MCTS) applied to Chinese Checkers. It features several enhancements and experimental refinements, including forward-move restrictions, GRAVE (Generalized Rapid Action Value Estimation), and heuristic/minimax-based playout policies. Due to high computational complexity and lengthy games, we used a simplified setting with a two-player variant. Base implementation is forked from https://github.com/Zedrichu/Chinese-Checkers-AI.git

## Repository Structure

- **`src/`**  
  Contains the source code for the game, search algorithms, and player types:
  - **`GameController.py`** – Manages game setup, player initialization, game loop, and tournament runs.
  - **`players/`** – Contains different player implementations:
    - `MCTSPlayer.py` – Classic MCTS with various enhancements.
    - `GRAVEPlayer.py`, `PPAPlayer.py`, `fwdMCTSPlayer.py` – Other variants incorporating GRAVE, policy adaptations, or forward-only constraints.
    - `MinimaxAIPlayer.py`, `RandomPlayer.py`, `NonRepeatRandomPlayer.py` – Baseline opponents.
    - `GraphicsHumanPlayer.py` – For interactive, GUI-based play.
  - **`game_problem/`** – Contains game-specific logic for Chinese Checkers:
    - `ChineseCheckers.py` – Defines board initialization, move legality, terminal states, etc.
    - `Heuristic.py` – Implements various heuristic evaluation functions (e.g., distance-based measures).
    - `GameProblem.py` – Common interfaces and definitions used throughout the project.
  - **`game/`** – Lower-level game definitions (e.g., `Board.py`, `State.py`, `Action.py`).

- **Tournament Runner**  
  A separate script `tournament.py` orchestrates repeated game simulations for evaluation. It tracks metrics such as game duration, turns played, winners, and prints sample entries from the transposition table for debugging.

## Running in Tournament Mode

   Use the provided command-line interface to specify player types and settings:

   ```bash
   python tournament.py --first-player PPA --second-player minimax --nb 1014 --num_games 200
