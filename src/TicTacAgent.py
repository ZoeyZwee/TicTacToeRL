"""
An agent to learn and play tic tac toe via self play.
"""
from TicTacToe import Board, GameStatus
import random
from collections import namedtuple
import numpy as np

StateValue = namedtuple("StateValue", ["value", "visits"])


class MonteCarloAgent:

    def __init__(self, player_id, board=None):
        """
        init a monte carlo agent. we have a model of the game, so we don't need an explicit policy
        :param player_id: 1 for X, 2 for O
        :param board: board the agent is playing on
        """
        self.game = Board() if board is None else board
        # keys are Boards, values are StateValues.
        self.value = {}  # lookup table for value fn. >0 means good for p1. <0 means good for p2
        self.player_id = player_id  # 1 if X, 2 if O

    def new_game(self, board=None):
        self.game = Board() if board is None else board

    def play_move(self, move):
        """
        Play a move and return who won (1 for p1, 2 for p2, 0 for game still on, -1 for draw
        :param move: position of move to be played (1-9)
        :return: 1 if p1 won. 2 if p2 won. 0 for game still going. -1 for draw.
        """
        return self.game.play_move(move, self.player_id)

    def play_random(self):
        legals = self.game.get_legals()
        return self.play_move(np.random.choice(legals))

    def get_best_move(self):
        """
        Simulate all legal moves, decide which is best for us.
        Positive value is good for p1.
        Negative value is good for p2
        :return: # from 1-9 indicating the best move
        """
        best_value = None
        best_move = None

        def compare(val):
            # see if val is better than the best move so far
            return val > best_value if self.player_id == 1 else val < best_value

        for move in self.game.get_legals():
            # simulate move to get value of resulting state
            next_state = self.game.sim_move(move, self.player_id)

            # add state to value fn on first visit
            if next_state not in self.value.keys():
                self.value[next_state] = StateValue(random.uniform(-1, 1), 0)

            move_value = self.value[next_state]

            # update best_value, best_move
            if best_value is None:
                best_value = move_value
                best_move = move
            elif compare(move_value):
                best_value = move_value
                best_move = move

        return best_move

    def update_value(self, states_visited, reward):
        for state in states_visited:
            if state not in self.value.keys():
                # sometimes moves by the other player don't get initialized in our value fn
                self.value[state] = StateValue(random.uniform(-1,1), 0)
            old_value = self.value[state].value
            visits = self.value[state].visits
            # incorporate trajectory into average
            new_value = (old_value*visits+reward)/(visits+1)
            self.value[state] = StateValue(new_value, visits+1)


if __name__ == "__main__":
    print()
