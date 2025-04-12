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



class TranspositionTableAMAF():
    def __init__(self, MaxLegalMoves, BoardLength) -> None:
        self.T={}
        self.MaxLegalMoves = MaxLegalMoves
        self.BoardLength = BoardLength
        self.CodeMaxLegalMoves = BoardLength**4 * 3

    def addAMAF(self, state:State):
        nplayouts = [0.0 for x in range (self.MaxLegalMoves)]
        nwins = [0.0 for x in range (self.MaxLegalMoves)]
        nplayoutsAMAF = [0.0 for x in range (self.CodeMaxLegalMoves)]
        nwinsAMAF = [0.0 for x in range (self.CodeMaxLegalMoves)]
        visited = [False for x in range(self.MaxLegalMoves)]
        self.T[hash(state)] = [0, nplayouts, nwins, nplayoutsAMAF, nwinsAMAF, visited]

    def updateAMAF(self, t, played, res):
        for i in range (len (played)):
            if played [:i].count (played [i]) == 0:
                t[3][played [i]] += 1
                t[4][played [i]] += res

    def look (self,state):
        return self.T.get(hash(state), None)
    
    def tocode(self, action):
        return (action.step_type-1) + 3 * ((action.src[0]* self.BoardLength + action.src[1]) * self.BoardLength**2 + \
                                           action.dest[0]* self.BoardLength + action.dest[1])



class GRAVEPlayer(Player):
    """
    Random player (confused AI) - selects an action randomly from the list of valid actions
    """

    def __init__(self, nb, MaxLegalMoves=60, BoardLength=7, min_visit=50):
        super().__init__()
        self._player_type = 'MCTS'
        self.T = TranspositionTableAMAF(MaxLegalMoves, BoardLength)
        self.nb = nb                    # number of playouts done before choosing a move
        self.min_visit = min_visit      # GRAVE hyperparameter

    def playoutAMAF(self, problem: GameProblem, state: State, played) -> int:
        new_state=state.copy()
        while not problem.terminal_test(new_state):
            action = random.choice(list(problem.actions(new_state)))
            new_state = problem.result(new_state, action)
            played.append(self.T.tocode(action))
        return problem.utility(new_state, state.player)
    
    @staticmethod
    def forward_playout(problem: GameProblem, state: State) -> int:
        new_state=state.copy()
        while not problem.terminal_test(new_state):
            moves = list(problem.forward_actions(new_state))
            
            if len(moves) == 0:
                moves = list(problem.actions(new_state))
            action = random.choice(moves)
            new_state = problem.result(new_state, action)
        return problem.utility(new_state, state.player)
    
    def searchGRAVE(self,  problem: GameProblem, state: State, played, tref):
        if problem.terminal_test(state):
            return problem.utility(state, state.player)
        t = self.T.look(state)
        if t != None:
            tr = tref
            if t[0]>self.min_visit:
                tr = t
            bestValue = 0
            best = 0
            moves = list(problem.actions(state))
            bestcode = self.T.tocode(moves[0])
            for i in range (0, len (moves)):
                val = 1000000.0
                code = self.T.tocode(moves[i])
                n = t[0]
                ni = t[1][i]
                wi = t[2][i]
                nAMAF = tr[3][code]
                wAMAF = tr[4][code]
                visited = t[5][i]
                if visited:
                    val = -val
                elif nAMAF > 0:
                    beta = nAMAF / (ni + nAMAF + 1e-5*ni*nAMAF)
                    Q = 1
                    if ni > 0:
                        Q = wi / ni
                        if state.player == 2:
                            Q = 1 - Q
                        # print(val)
                    AMAF = wAMAF / nAMAF
                    if state.player == 2:
                        AMAF = 1 - AMAF
                    val = (1.0 - beta)*Q + beta*AMAF

                if val > bestValue:
                    bestValue = val
                    best = i
                    bestcode = code
            # print("#######################")
            # print(bestValue)
            # print(state.board)
            state = problem.result(state, moves[best])
            played.append(bestcode)
            # print(state.board)
            # print(moves)
            t[3][best] = True                   # Useless to visit same node twice in a same descent
            res = self.searchGRAVE(problem, state, played, tr)
            t[3][best] = False
            t [0] += 1
            t [1] [best] += 1
            t [2] [best] += res
            self.T.updateAMAF(t, played, res)
            return res
        else:
            self.T.addAMAF(state)
            return self.playoutAMAF(problem, state, played)

    def get_action(self, problem: GameProblem, state: State) -> Action:
        self.T.addAMAF(state)
        root = self.T.look(state)
        for i in range(self.nb):
            s1 = state.copy()
            res = self.searchGRAVE(problem, s1, [], root)
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

