from math import log, sqrt
import random
import time
import sys

if __name__=="__main__":
    sys.path.append("src")

from game_problem.ChineseCheckers import ChineseCheckers

from game_problem.GameProblem import GameProblem
from game.Action import Action
from game.State import State
from players.Player import Player



class TranspositionTable():
    def __init__(self, MaxLegalMoves) -> None:
        self.T={}
        self.MaxLegalMoves = MaxLegalMoves

    def add (self, state:State):
        nplayouts = [0.0 for x in range (self.MaxLegalMoves)]
        nwins = [0.0 for x in range (self.MaxLegalMoves)]
        visited = [False for x in range(self.MaxLegalMoves)]
        self.T[hash(state)] = [0, nplayouts, nwins, visited]

    def look (self,state):
        return self.T.get(hash(state), None)



class MCTSPlayer(Player):
    """
    Random player (confused AI) - selects an action randomly from the list of valid actions
    """

    def __init__(self, nb, MaxLegalMoves=60):
        super().__init__()
        self._player_type = 'MCTS'
        self.T = TranspositionTable(MaxLegalMoves)
        self.nb = nb                    # number of playouts done before choosing a move

    @staticmethod
    def playout(problem: GameProblem, state: State) -> int:
        new_state=state.copy()
        while not problem.terminal_test(new_state):
            action = random.choice(list(problem.actions(new_state)))
            new_state = problem.result(new_state, action)
        return (problem.utility(new_state, state.player) + 1) / 2
    
    @staticmethod
    def forward_playout(problem: GameProblem, state: State) -> int:
        new_state=state.copy()
        while not problem.terminal_test(new_state):
            moves = list(problem.forward_actions(new_state))
            
            if len(moves) == 0:
                moves = list(problem.actions(new_state))
            action = random.choice(moves)
            new_state = problem.result(new_state, action)
        return (problem.utility(new_state, state.player) + 1) / 2
    
    def search(self,  problem: GameProblem, state: State):
        if problem.terminal_test(state):
            return (problem.utility(state, state.player) + 1) / 2
        t = self.T.look(state)
        if t != None:
            bestValue = 0
            best = 0
            moves = list(problem.actions(state))
            for i in range (0, len (moves)):
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
                    val = Q + 0.4 * sqrt (log (n) / ni)
                    # print(val)
                if val > bestValue:
                    bestValue = val
                    best = i
            # print("#######################")
            # print(bestValue)
            # print(state.board)
            state = problem.result(state, moves[best])
            # print(state.board)
            # print(moves)
            t[3][best] = True                   # Useless to visit same node twice in a same descent
            res = self.search(problem, state)
            t[3][best] = False
            t [0] += 1
            t [1] [best] += 1
            t [2] [best] += res
            return res
        else:
            self.T.add(state)
            return MCTSPlayer.forward_playout(problem, state)

    def get_action(self, problem: GameProblem, state: State) -> Action:
        for i in range(self.nb):
            s1 = state.copy()
            res = self.search(problem, s1)
        t = self.T.look (state)
        moves = list(problem.actions(state))
        best = moves [0]
        bestValue = t [1] [0]
        for i in range (1, len(moves)):
            if (t [1] [i] > bestValue):
                bestValue = t [1] [i]
                best = moves [i]
        self._moves_count += 1
        return best


if __name__=="__main__":
    mcts = MCTSPlayer(2,3)
    problem = ChineseCheckers(triangle_size=3)
    state = problem.initial_state()
    start = time.time()
    for i in range(1):
        res = MCTSPlayer.forward_playout(problem,state)
    print(res)
    print(time.time()-start)