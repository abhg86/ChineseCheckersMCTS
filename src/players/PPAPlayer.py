import copy
from math import log, sqrt
import math
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
    def __init__(self, MaxLegalMoves, BoardLength=7) -> None:
        self.T={}
        self.MaxLegalMoves = MaxLegalMoves
        self.BoardLength = BoardLength
        self.CodeMaxLegalMoves = BoardLength**4 * 3

    def add (self, state:State):
        nplayouts = [0.0 for x in range (self.MaxLegalMoves)]
        nwins = [0.0 for x in range (self.MaxLegalMoves)]
        visited = [False for x in range(self.MaxLegalMoves)]
        self.T[hash(state)] = [0, nplayouts, nwins, visited]

    def look (self,state):
        return self.T.get(hash(state), None)

    def tocode(self, action):
        return (action.step_type-1) + 3 * ((action.src[0]* self.BoardLength + action.src[1]) * self.BoardLength**2 + \
                                           action.dest[0]* self.BoardLength + action.dest[1])

class Policy():
    def __init__(self):
        self.p = {}
    
    def put(self, move_code, value):
        if move_code in self.p:
            self.p[move_code] += value
        else:
            self.p[move_code] = value
    
    def get(self, move_code, uniform_value):
        if move_code in self.p:
            return self.p[move_code]
        return uniform_value

class PPAPlayer(Player):
    """
    Random player (confused AI) - selects an action randomly from the list of valid actions
    """

    def __init__(self, nb, player, MaxLegalMoves=60):
        super().__init__()
        self._player_type = 'PPA'
        self.T = TranspositionTable(MaxLegalMoves)
        self.P = Policy()
        self.nb = nb                    # number of playouts done before choosing a move
        self.player = player

    def playout(self,problem: GameProblem, state: State) -> int:
        new_state=state.copy()
        played = []
        while not problem.terminal_test(new_state):
            moves = list(problem.forward_actions(new_state))
            if len(moves) == 0:
                moves = list(problem.actions(new_state))
            z=0
            for move in moves:
                z += math.exp(self.P.get(self.T.tocode(move), 1/len(moves)))
            stop = random.random() * z
            move = 0
            z=0
            while True:
                z += math.exp(self.P.get(self.T.tocode(moves[move]), 1/len(moves)))
                if z>= stop:
                    break
                move += 1
            
            new_state = problem.result(new_state, moves[move])
            played.append(moves[move])
        return (problem.utility(new_state, state.player) + 1) / 2, played
    
    def adapt(self, problem, winner, player, state, playout):
        polp = copy.deepcopy(self.P)
        new_state = state.copy()
        alpha = 0.32
        for move in playout:
            if player == winner:
                polp.put(self.T.tocode(move), alpha)
                z=0
                possible_moves = list(problem.actions(new_state))

                for possible_move in possible_moves:
                    z += math.exp(self.P.get(self.T.tocode(possible_move), 1/len(possible_moves)))

                for possible_move in possible_moves:
                    polp.put(self.T.tocode(possible_move), - alpha * math.exp(self.P.get(self.T.tocode(possible_move), 1/len(possible_moves)))/z)
            new_state = problem.result(new_state, move)
            player = 3 - player
        self.P = polp
    
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
            t [2] [best] += res[0]
            return res
        else:
            self.T.add(state)
            return self.playout(problem, state)

    def get_action(self, problem: GameProblem, state: State) -> Action:
        for i in range(self.nb):
            s1 = state.copy()
            res = self.search(problem, s1)
            if res[0] == 1:
                self.adapt(problem,1, self.player, s1,res[1])
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

