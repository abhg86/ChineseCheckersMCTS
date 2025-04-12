from math import log, sqrt
import random
import time
import sys
import math

if __name__ == "__main__":
    sys.path.append("src")

from game_problem.ChineseCheckers import ChineseCheckers
from game_problem.Heuristic import *
from game_problem.GameProblem import GameProblem
from game.Action import Action
from game.State import State
from players.Player import Player


class TranspositionTable:
    def __init__(self, MaxLegalMoves) -> None:
        self.T = {}
        self.MaxLegalMoves = MaxLegalMoves

    def add(self, state: State):
        nplayouts = [0.0 for x in range(self.MaxLegalMoves)]
        nwins = [0.0 for x in range(self.MaxLegalMoves)]
        visited = [False for x in range(self.MaxLegalMoves)]
        self.T[hash(state)] = [0, nplayouts, nwins, visited]

    def look(self, state):
        return self.T.get(hash(state), None)


def minimax(heuristic, problem, state, depth):
    # If we've reached the depth limit or a terminal state, return an evaluation.
    if problem.terminal_test(state):
        # Use the problem's utility function as the evaluation,
        # or a more nuanced heuristic evaluation.
        return problem.utility(state, state.player)
    elif depth == 0:
        return heuristic.eval(state, state.player)

    if state.player == 1:
        best_val = -math.inf
        for action in problem.actions(state):
            new_state = problem.result(state, action)
            val = minimax(heuristic, problem, new_state, depth - 1)
            best_val = max(best_val, val)
        return best_val
    else:
        best_val = math.inf
        for action in problem.actions(state):
            new_state = problem.result(state, action)
            val = minimax(heuristic, problem, new_state, depth - 1)
            best_val = min(best_val, val)
        return best_val


class MCTSPlayer(Player):

    def __init__(self, nb, heuristic: Heuristic, MaxLegalMoves=60):
        super().__init__()
        self._player_type = "MCTS"
        self.T = TranspositionTable(MaxLegalMoves)
        self.nb = nb  # number of playouts per move
        self.force_forward_moves = False  # New flag: if True, use only forward moves
        self.use_minimax_rollout = True  # New flag: if True, use minimax for playouts
        self.heuristic = heuristic

    @staticmethod
    def playout(problem: GameProblem, state: State) -> int:
        new_state = state.copy()
        while not problem.terminal_test(new_state):
            action = random.choice(list(problem.actions(new_state)))
            new_state = problem.result(new_state, action)
        return problem.utility(new_state, state.player)

    @staticmethod
    def forward_playout(problem: GameProblem, state: State) -> int:
        new_state = state.copy()
        while not problem.terminal_test(new_state):
            moves = list(problem.forward_actions(new_state))

            if len(moves) == 0:
                moves = list(problem.actions(new_state))
            action = random.choice(moves)
            new_state = problem.result(new_state, action)
        return problem.utility(new_state, state.player)

    @staticmethod
    def minimax_playout(self, problem: GameProblem, state: State) -> int:
        # Define a depth limit for minimax simulation; you can tune this value.
        MINIMAX_DEPTH = 3
        # Optionally, check a flag to decide whether to use minimax heuristic:
        use_minimax = self.use_minimax_rollout
        if use_minimax:
            return minimax(self.heuristic, problem, state, MINIMAX_DEPTH)
        else:
            # Fall back to pure random simulation if not using minimax.
            new_state = state.copy()
            while not problem.terminal_test(new_state):
                moves = list(problem.forward_actions(new_state))
                if len(moves) == 0:
                    moves = list(problem.actions(new_state))
                action = random.choice(moves)
                new_state = problem.result(new_state, action)
            return problem.utility(new_state, state.player)

    def search(self, problem: GameProblem, state: State):
        if problem.terminal_test(state):
            return problem.utility(state, state.player)
        t = self.T.look(state)
        # Use forward_actions if forcing forward moves; otherwise use all legal actions.
        if self.force_forward_moves:
            moves = list(problem.forward_actions(state))
        else:
            moves = list(problem.actions(state))
        if t is not None:
            bestValue = 0
            best = 0
            for i in range(len(moves)):
                val = 1000000.0
                n = t[0]
                ni = t[1][i]
                wi = t[2][i]
                visited = t[3][i]
                if visited:
                    val = -val
                elif ni > 0:
                    Q = wi / ni
                    if state.player == 2:
                        Q = 1 - Q
                    val = Q + 0.4 * sqrt(log(n) / ni)
                if val > bestValue:
                    bestValue = val
                    best = i
            state = problem.result(state, moves[best])
            t[3][best] = True
            res = self.search(problem, state)
            t[3][best] = False
            t[0] += 1
            t[1][best] += 1
            t[2][best] += res
            return res
        else:
            self.T.add(state)
            return MCTSPlayer.minimax_playout(self, problem, state)

    def get_action(self, problem: GameProblem, state: State) -> Action:
        for i in range(self.nb):
            s1 = state.copy()
            _ = self.search(problem, s1)
        # When choosing the final move, also use forward moves if forced.
        if self.force_forward_moves:
            moves = list(problem.forward_actions(state))
        else:
            moves = list(problem.actions(state))
        t = self.T.look(state)
        best = moves[0]
        bestValue = t[1][0]
        for i in range(1, len(moves)):
            if t[1][i] > bestValue:
                bestValue = t[1][i]
                best = moves[i]
        self._moves_count += 1
        return best


if __name__ == "__main__":
    mcts = MCTSPlayer(2, 3)
    problem = ChineseCheckers(triangle_size=3)
    state = problem.initial_state()
    start = time.time()
    for i in range(1):
        res = MCTSPlayer.forward_playout(problem, state)
    print(res)
    print(time.time() - start)
