import argparse
import sys

sys.path.append("src")
from GameController import GameController


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chinese Checkers game with AI and player options.')

    parser.add_argument('--first-player', choices=['human', 'minimax','random', 'nonrepeatrandom', 'MCTS', 'GRAVE', 'PPA'], required=False,
                        help='Type of the first player.')
    parser.add_argument('--first-minimax-depth', type=int, default=6, required=False,
                        help='Minimax depth for the first player, if applicable.')
    parser.add_argument('--second-player', choices=['minimax','random', 'nonrepeatrandom', 'MCTS', 'GRAVE', 'PPA'], required=False,
                        help='Type of the second player.')
    parser.add_argument('--second-minimax-depth', type=int, default=6, required=False,
                        help='Minimax depth for the second player, if applicable.')
    parser.add_argument('--no_gui', required=False, action='store_false',
                        help="Graphical interface")
    parser.add_argument('--nb', type=int, default=512, required=False,
                        help='Number of playouts done by MCTS')

    args = parser.parse_args()

    controller = GameController(verbose=False, use_graphics=args.no_gui, args=args)

    controller.game_loop()
