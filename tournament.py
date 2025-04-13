import argparse
import sys
import time
import json

sys.path.append("src")
from GameController import GameController


class Tournament:
    def __init__(self, controller, num_games):
        self.controller = controller  # This controller is created once so that player objects (and MCTS table) persist.
        self.num_games = num_games
        self.results = []  # List to hold per-game metrics.

    def run(self):
        wins = {1: 0, 2: 0}
        total_turns = 0
        total_duration = 0.0
        p1_times = []  # To accumulate average turn times for player 1.
        p2_times = []  # To accumulate average turn times for player 2.

        for i in range(1, self.num_games + 1):
            # For the first 50 games, force forward moves:
            if i <= 50:
                self.controller.players[0].force_forward_moves = True
            else:
                self.controller.players[0].force_forward_moves = False
            # run_single_game() is assumed to return a dict with keys:
            # "winner", "turns", "duration", "avg_time_p1", "avg_time_p2", "final_state"
            game_data = self.controller.run_single_game()
            duration = game_data["duration"]
            winner = game_data["winner"]
            turns = game_data["turns"]
            avg_time_p1 = game_data.get("avg_time_p1", None)
            avg_time_p2 = game_data.get("avg_time_p2", None)
            final_state = game_data.get("final_state", None)

            wins[winner] += 1
            total_turns += turns
            total_duration += duration
            if avg_time_p1 is not None:
                p1_times.append(avg_time_p1)
            if avg_time_p2 is not None:
                p2_times.append(avg_time_p2)

            print(
                f"Game {i}: Winner = Player {winner}, Turns = {turns}, Duration = {duration:.3f} sec, "
                f"Final state: {final_state}"
            )
            self.results.append(game_data)

            table = self.controller.players[0].T.T

            # Print a few sample state entries.
            sample_entries = list(table.items())[:1]
            for idx, (state_hash, entry) in enumerate(sample_entries):
                total_visits = entry[0]
                move_visits = entry[1]
                wins = entry[2]
                # You might also print the visited flags (entry[3]) if needed.
                print(f"\nSample Entry {idx + 1}:")
                print(f"State hash: {state_hash}")
                print(f"  Total visits: {total_visits}")
                print(f"  Move visits: {move_visits}")
                print(f"  Win totals  : {wins}")

        avg_turns = total_turns / self.num_games if self.num_games > 0 else 0
        avg_duration = total_duration / self.num_games if self.num_games > 0 else 0
        overall_avg_p1 = sum(p1_times) / len(p1_times) if p1_times else None
        overall_avg_p2 = sum(p2_times) / len(p2_times) if p2_times else None

        print("\nTournament Results:")
        print(f"Player 1 wins (MCTS): {wins[1]}")
        print(f"Player 2 wins (Minimax): {wins[2]}")
        print(f"Average turns per game: {avg_turns:.2f}")
        print(f"Average duration per game: {avg_duration:.3f} sec")
        print(f"Overall average time per turn, Player 1: {overall_avg_p1}")
        print(f"Overall average time per turn, Player 2: {overall_avg_p2}")

    def save_results(self, filename):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chinese Checkers Tournament")
    parser.add_argument(
        "--first-player",
        choices=["human", "minimax", "random", "nonrepeatrandom", "MCTS", "GRAVE"],
        default="MCTS",
        help="Type of the first player.",
    )
    parser.add_argument(
        "--first-minimax-depth",
        type=int,
        default=6,
        help="Minimax depth for the first player, if applicable.",
    )
    parser.add_argument(
        "--second-player",
        choices=["minimax", "random", "nonrepeatrandom", "MCTS", "GRAVE"],
        default="minimax",
        help="Type of the second player.",
    )
    parser.add_argument(
        "--second-minimax-depth",
        type=int,
        default=6,
        help="Minimax depth for the second player, if applicable.",
    )
    parser.add_argument(
        "--no_gui", action="store_false", help="Disable Graphical interface"
    )
    parser.add_argument(
        "--nb", type=int, default=8, help="Number of playouts done by MCTS"
    )
    parser.add_argument(
        "--num_games", type=int, default=200, help="Number of games in the tournament"
    )
    args = parser.parse_args()

    # Create the controller instance; it builds the players using args.
    controller = GameController(verbose=False, use_graphics=args.no_gui, args=args)

    # Create and run the tournament.
    tournament = Tournament(controller, args.num_games)
    tournament.run()

    # Build filename as: firststrategy_secondstrategy_nbofplayouts(nb).json
    filename = f"{args.first_player}_{args.second_player}_{args.nb}.json"
    tournament.save_results(filename)
