import time

# from benchmarking.GameAnalytics import GameAnalytics
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
from players.PPAPlayer import PPAPlayer
from players.RandomPlayer import RandomPlayer
from game_problem.ChineseCheckers import ChineseCheckers
from game.Graphics import Graphics
from players.fwdMCTSPlayer import fwdMCTSPlayer
from utils import play_beep


def create_player(
    player_type,
    player_nb,
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
        return MCTSPlayer(nb, player_nb)
    elif player_type == "GRAVE":
        return GRAVEPlayer(nb, player_nb)
    elif player_type == "PPA":
        return PPAPlayer(nb, player_nb)
    elif player_type == "fwdMCTS":
        return fwdMCTSPlayer(nb, player_nb)
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
        # self.analytics = GameAnalytics()
        self.verbose = verbose  # Flag to print the state and action applied
        self.use_graphics = use_graphics  # Flag to use the GUI
        self.problem = ChineseCheckers(triangle_size=3)  # Initialize the game problem
        self.gui = (
            Graphics() if use_graphics else None
        )  # Initialize the GUI if the flag is set
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

        if args.first_player is None or args.second_player is None:
            # self.players = build_test_subject_euclidean_vs_euclidean_each_corner(self.problem, 6, 6, self.verbose)
            # self.players = build_test_subject_euclidean_vs_manhattan(self.problem, 6, 6, self.verbose)
            # self.players = build_test_subject_manhattan_vs_manhattan_each_corner(self.problem, 5, 5, self.verbose)
            # self.players = build_test_subject_weighted_single_corner_vs_weighted_each_corners(self.problem, 6, 6, self.verbose)
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
                    args.first_player in ["MCTS", "GRAVE", "PPA", "fwdMCTS"]
                    or args.second_player in ["MCTS", "GRAVE", "PPA", "fwdMCTS"]
                )
                else None
            )
            player1 = create_player(
                args.first_player,
                1,
                depth=player1_depth,
                gui=self.gui,
                problem=self.problem,
                max_player=1,
                heuristic=default_heuristic,
                nb=nb,
            )
            player2 = create_player(
                args.second_player,
                2,
                depth=player2_depth,
                gui=self.gui,
                problem=self.problem,
                max_player=2,
                heuristic=default_heuristic,
                nb=nb,
            )
            self.players.append(player1)
            self.players.append(player2)

    def game_loop(self):
        """
        Main game loop - takes care of the game state and the players' turns.
        """
        # Initialize the game state with the game problem definition
        state = self.problem.initial_state()
        # Start the game timer to measure the game duration and other performance metrics
        game_start_timer = time.perf_counter()

        turn = 0
        # Loop until the game is over - state becomes terminal
        while not self.problem.terminal_test(state):
            # Retrieve the action from the player currently taking the turn
            action = self.players[state.player - 1].get_action(self.problem, state)
            # Update the state according to the action applied
            state = self.problem.result(state, action)

            # Print the state and action applied once every 10 turns
            if self.verbose:
                print(
                    f"Player {state.player} | applied action: {action} | turn = {turn}"
                )
                print(state)
                print("\n\n")

            # Connection to the GUI
            if self.gui:
                self.gui.handle_quit()
                self.gui.draw_everything(state)

            turn += 1

        # Print the final state and the utility of the final state
        print(f"Final state:\n{state}")

        game_duration = time.perf_counter() - game_start_timer
        print(
            f"Player {state.player} has utility: {self.problem.utility(state, state.player)}"
        )

        winner = (
            state.player
            if self.problem.utility(state, state.player) >= 0
            else len(self.players) + 1 - state.player
        )

        # Print the game duration and the performance metrics of the players
        # self.analytics.add_game_data(game_duration, turn, self.players, winner)
        # self.analytics.print_game_data()
        # self.analytics.save_to_file()

        # code used to plot the results
        # self.analytics.load_from_file('game_data.json')
        # self.analytics.plot()

        play_beep()

        # Wait until quit is pressed
        if self.gui:
            while True:
                self.gui.handle_quit()

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
