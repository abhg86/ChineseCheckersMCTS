import time

from game_problem.GameProblem import GameProblem
from players.GRAVEPlayer import GRAVEPlayer
from players.MCTSPlayer import MCTSPlayer
from players.NonRepeatRandomPlayer import NonRepeatingRandomPlayer
from game_problem.Heuristic import (
    WeightedHeuristic,
    SumOfPegsInCornerHeuristic,
    AverageManhattanToCornerHeuristic,
    AverageEuclideanToCornerHeuristic,
    MaxManhattanToCornerHeuristic,
    EnsuredNormalizedHeuristic,
    AverageEuclideanToEachCornerHeuristic,
    AverageManhattanToEachCornerHeuristic,
)
from players.GraphicsHumanPlayer import GraphicsHumanPlayer
from players.MinimaxAIPlayer import MinimaxAIPlayer
from players.RandomPlayer import RandomPlayer
from game_problem.ChineseCheckers import ChineseCheckers
from game.Graphics import Graphics
from utils import play_beep


def create_player(
    player_type,
    depth=6,
    gui=None,
    problem=None,
    max_player=None,
    heuristic=None,
    nb=None,
):
    if player_type == "human":
        return GraphicsHumanPlayer(gui)
    elif player_type == "random":
        return RandomPlayer()
    elif player_type == "nonrepeatrandom":
        return NonRepeatingRandomPlayer()
    elif player_type == "minimax":
        return MinimaxAIPlayer(
            problem, max_player, max_depth=depth, heuristic=heuristic, verbose=False
        )
    elif player_type == "MCTS":
        return MCTSPlayer(nb, heuristic=heuristic)
    elif player_type == "GRAVE":
        return GRAVEPlayer(nb)
    else:
        raise ValueError("Unsupported player type")


def build_test_subject_with_default_weighted_heuristic(
    problem: GameProblem, depth: int, verbose=False
):
    heuristic1 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.1),
            (AverageManhattanToCornerHeuristic(), 0.3),
            (AverageEuclideanToCornerHeuristic(), 0.4),
            (MaxManhattanToCornerHeuristic(), 0.2),
        ]
    )
    return [
        MinimaxAIPlayer(problem, 1, depth, heuristic1, verbose=verbose),
        MinimaxAIPlayer(problem, 2, depth, heuristic1, verbose=verbose),
    ]


def build_test_subject_euclidean_vs_manhattan(
    problem: GameProblem, depth1: int, depth2: int, verbose=False
):
    heuristic1 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.2),
            (AverageEuclideanToCornerHeuristic(), 0.8),
        ]
    )

    heuristic2 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.2),
            (AverageManhattanToCornerHeuristic(), 0.8),
        ]
    )
    return [
        MinimaxAIPlayer(
            problem, 1, depth1, heuristic1, verbose=verbose, title="AverageEuclidean"
        ),
        MinimaxAIPlayer(
            problem, 2, depth2, heuristic2, verbose=verbose, title="AverageManhattan"
        ),
        # MinimaxAIPlayer(problem, 1, depth2, heuristic2, verbose=verbose, title='AverageManhattan'),
        # MinimaxAIPlayer(problem, 2, depth1, heuristic1, verbose=verbose, title='AverageEuclidean'),
    ]


def build_test_subject_euclidean_vs_euclidean_each_corner(
    problem: GameProblem, depth1: int, depth2: int, verbose=False
):
    heuristic1 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.5),
            (AverageEuclideanToCornerHeuristic(), 0.5),
        ]
    )

    heuristic2 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.5),
            (AverageEuclideanToEachCornerHeuristic(), 0.5),
        ]
    )
    return [
        # MinimaxAIPlayer(problem, 1, depth2, heuristic2, verbose=verbose, title='AverageEuclideanToEachCorner'),
        # MinimaxAIPlayer(problem, 2, depth1, heuristic1, verbose=verbose, title='AverageEuclidean'),
        MinimaxAIPlayer(
            problem, 1, depth1, heuristic1, verbose=verbose, title="AverageEuclidean"
        ),
        MinimaxAIPlayer(
            problem,
            2,
            depth2,
            heuristic2,
            verbose=verbose,
            title="AverageEuclideanToEachCorner",
        ),
    ]


def build_test_subject_manhattan_vs_manhattan_each_corner(
    problem: GameProblem, depth1: int, depth2: int, verbose=False
):
    heuristic1 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.2),
            (AverageManhattanToCornerHeuristic(), 0.8),
        ]
    )

    heuristic2 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.2),
            (AverageManhattanToEachCornerHeuristic(), 0.8),
        ]
    )
    return [
        MinimaxAIPlayer(
            problem, 1, depth1, heuristic1, verbose=verbose, title="AverageManhattan"
        ),
        MinimaxAIPlayer(
            problem,
            2,
            depth2,
            heuristic2,
            verbose=verbose,
            title="AverageManhattanToEachCorner",
        ),
        # MinimaxAIPlayer(problem, 1, depth2, heuristic2, verbose=verbose, title='AverageManhattanToEachCorner'),
        # MinimaxAIPlayer(problem, 2, depth1, heuristic1, verbose=verbose, title='AverageManhattan'),
    ]


def build_test_subject_weighted_single_corner_vs_weighted_each_corners(
    problem: GameProblem, depth1: int, depth2: int, verbose=False
):
    heuristic1 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.5),
            (AverageManhattanToCornerHeuristic(), 0.2),
            (AverageEuclideanToCornerHeuristic(), 0.2),
            (MaxManhattanToCornerHeuristic(), 0.1),
        ]
    )

    heuristic2 = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.5),
            (AverageManhattanToEachCornerHeuristic(), 0.2),
            (AverageEuclideanToEachCornerHeuristic(), 0.2),
            (MaxManhattanToCornerHeuristic(), 0.1),
        ]
    )
    return [
        MinimaxAIPlayer(
            problem,
            1,
            depth1,
            heuristic1,
            verbose=verbose,
            title="WeightedSingleCorner",
        ),
        MinimaxAIPlayer(
            problem, 2, depth2, heuristic2, verbose=verbose, title="WeightedEachCorner"
        ),
        # MinimaxAIPlayer(problem, 1, depth2, heuristic2, verbose=verbose, title='WeightedEachCorner'),
        # MinimaxAIPlayer(problem, 2, depth1, heuristic1, verbose=verbose, title='WeightedSingleCorner'),
    ]


def build_test_subject_both_with_weighted_each_corners(
    problem: GameProblem, depth: int, verbose=False
):
    heuristic = WeightedHeuristic(
        [
            (SumOfPegsInCornerHeuristic(), 0.1),
            (AverageManhattanToEachCornerHeuristic(), 0.3),
            (AverageEuclideanToEachCornerHeuristic(), 0.4),
            (MaxManhattanToCornerHeuristic(), 0.2),
        ]
    )
    return [
        MinimaxAIPlayer(
            problem, 1, depth, heuristic, verbose=verbose, title="WeightedEachCorner"
        ),
        MinimaxAIPlayer(
            problem, 2, depth, heuristic, verbose=verbose, title="WeightedEachCorner"
        ),
    ]


class GameController:
    def __init__(self, verbose=True, use_graphics=True, args=None):
        self.verbose = verbose  # Flag to print state and actions
        self.use_graphics = use_graphics  # Flag to use the GUI
        self.problem = ChineseCheckers(triangle_size=3)  # Initialize the game problem
        self.gui = Graphics() if use_graphics else None  # Initialize the GUI if needed
        self.players = []
        self.handle_game_setup(args)

    def handle_game_setup(self, args):
        default_heuristic = WeightedHeuristic(
            [
                (SumOfPegsInCornerHeuristic(), 0.1),
                (AverageManhattanToCornerHeuristic(), 0.3),
                (AverageEuclideanToCornerHeuristic(), 0.4),
                (MaxManhattanToCornerHeuristic(), 0.2),
            ]
        )

        if args is None or args.first_player is None or args.second_player is None:
            # Use default test subjects if no player types are provided.
            self.players = build_test_subject_both_with_weighted_each_corners(
                self.problem, 6, self.verbose
            )
        else:
            player1_depth = (
                args.first_minimax_depth if args.first_player == "minimax" else None
            )
            player2_depth = (
                args.second_minimax_depth if args.second_player == "minimax" else None
            )
            nb = (
                args.nb
                if (
                    args.first_player in ["MCTS", "GRAVE"]
                    or args.second_player in ["MCTS", "GRAVE"]
                )
                else None
            )
            player1 = create_player(
                args.first_player,
                depth=player1_depth,
                gui=self.gui,
                problem=self.problem,
                max_player=1,
                heuristic=default_heuristic,
                nb=nb,
            )
            player2 = create_player(
                args.second_player,
                depth=player2_depth,
                gui=self.gui,
                problem=self.problem,
                max_player=2,
                heuristic=default_heuristic,
                nb=nb,
            )
            self.players.append(player1)
            self.players.append(player2)

    def run_single_game(self):
        """
        Runs a single game until a terminal state is reached.
        Returns a dictionary with:
          - "winner": the winning player's number (1 or 2)
          - "turns": total number of moves taken in the game
          - "duration": total game duration in seconds
          - "avg_time_p1": average time per turn for player 1
          - "avg_time_p2": average time per turn for player 2
          - "final_state": a string representation of the final board state
        """
        state = self.problem.initial_state()
        turn = 0
        t_game_start = time.perf_counter()

        total_time_p1 = 0.0
        count_p1 = 0
        total_time_p2 = 0.0
        count_p2 = 0

        # Play until the game reaches a terminal state.
        while not self.problem.terminal_test(state):
            current_player = state.player  # current player's number (1 or 2)

            # Measure time spent for this turn.
            t_turn_start = time.perf_counter()
            action = self.players[current_player - 1].get_action(self.problem, state)
            turn_time = time.perf_counter() - t_turn_start

            # Accumulate timing data per player.
            if current_player == 1:
                total_time_p1 += turn_time
                count_p1 += 1
            else:
                total_time_p2 += turn_time
                count_p2 += 1

            # Update state based on the chosen action.
            state = self.problem.result(state, action)

            if self.verbose:
                print(f"Turn {turn}: Player {current_player} applied action: {action}")
                print(state, "\n")

            if self.gui:
                self.gui.handle_quit()
                self.gui.draw_everything(state)

            turn += 1

        # Total game duration.
        duration = time.perf_counter() - t_game_start

        # Determine the winner based on the utility function.
        if self.problem.utility(state, state.player) >= 0:
            winner = state.player
        else:
            winner = len(self.players) + 1 - state.player

        # Compute average time per turn for each player.
        avg_time_p1 = total_time_p1 / count_p1 if count_p1 > 0 else 0
        avg_time_p2 = total_time_p2 / count_p2 if count_p2 > 0 else 0

        # Get a string representation of the final board state.
        final_state_str = str(state.board)

        # Build and return the game metrics dictionary.
        return {
            "winner": winner,
            "turns": turn,
            "duration": duration,
            "avg_time_p1": avg_time_p1,
            "avg_time_p2": avg_time_p2,
            "final_state": final_state_str,
        }

    def game_loop(self):
        """
        Interactive game loop that runs until the game is over.
        (This method is kept for interactive play; for tournament mode, use run_single_game().)
        """
        state = self.problem.initial_state()
        t_game_start = time.perf_counter()
        turn = 0

        while not self.problem.terminal_test(state):
            current_player = state.player
            action = self.players[current_player - 1].get_action(self.problem, state)
            state = self.problem.result(state, action)

            if self.verbose:
                print(f"Turn {turn}: Player {current_player} applied action: {action}")
                print(state, "\n")

            if self.gui:
                self.gui.handle_quit()
                self.gui.draw_everything(state)
            turn += 1

        duration = time.perf_counter() - t_game_start
        print(f"Final state:\n{state}")
        print(f"Game finished in {turn} turns, duration: {duration:.3f} sec")

        if self.problem.utility(state, state.player) >= 0:
            winner = state.player
        else:
            winner = len(self.players) + 1 - state.player
        print(f"Winner is Player {winner}")

        # In interactive mode you might want to wait for GUI quit.
        if self.gui:
            while True:
                self.gui.handle_quit()
