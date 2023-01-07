from TicTacToe import Board, GameStatus
from Agent import *
import json
from Plotter import Plotter
import numpy as np


def human_v_human():
    """
    Play human vs human in the console
    """
    def get_move(player):
        """
        Read move from console input
        """
        x = ""
        ins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        while x not in ins:
            try:
                x = int(input(f"Player {player}: "))
            except Exception:
                pass
        return x

    tt = Board()
    p1 = True
    print("A new game of tic tac toe wow!!!!")
    print("Valid inputs are 1-9")
    status = GameStatus.RUNNING
    while status == GameStatus.RUNNING:
        player = 1 if p1 else 2
        status = tt.play_move(get_move(player), player)
        p1 = not p1
        print(tt)
    print(f"Player {status} has won!!!!")
    print("If player 4 won that means a draw and you both suck")


def __train_agents(p1, p2, p1_value_path, p2_value_path):
    """
    train two agents by playing 500,000 matches
    p1 and p2 could theoretically be different agent types, though I haven't tested it yet
    :param p1: Agent playing X
    :param p2: Agent playing Y
    :param p1_value_path: where to save value fn of p1
    :param p2_value_path: where to save value fn of p2
    :return:
    """

    # learning params
    epsilon = 0.01  # % chance of a random move
    gamma = 0.90  # decay rate for rewards
    alpha = 0.01  # learning rate
    games_played = 0

    plotter = Plotter(p1, p2)

    startermove = 1

    plot = False  # only plot if we've seen all the start states (plotter code breaks otherwise)
    while games_played < 500_000:

        # start with each opener evenly
        match startermove:
            case 1:
                startermove = 2
            case 2:
                startermove = 5
            case 5:
                startermove = 1
                plot = True  # all start states seen - now safe to plot
        # play match
        outcome, game_log = play_match(p1, p2, epsilon, startermove)
        games_played += 1

        # update value fns
        # p1 trains on all its "afterstates", p2 on its "afterstates".
        p1.train(game_log[::2], REWARDS[outcome], gamma=gamma, alpha=alpha)
        p2.train(game_log[1::2], REWARDS[outcome], gamma=gamma, alpha=alpha)

        # logging and plotting
        if plot:
            openers = {1: "Corner", 2: "Side", 5: "Centre"}
            plotter.log_and_plot(outcome, opener=openers[startermove])

        # save value fn
        if games_played % 1000 == 0:
            print(f"saving value functions... ({len(p1.value) + len(p2.value)} states seen, {games_played} games played)")
            with open(p1_value_path, "w") as f:
                d = {str(board): val for board, val in p1.value.items()}
                json.dump(d, f)
            with open(p2_value_path, "w") as f:
                d = {str(board): val for board, val in p2.value.items()}
                json.dump(d, f)


def train_TD():
    p1 = TDAgent(1)
    p2 = TDAgent(2)
    __train_agents(p1, p2, "TDValueX.json", "TDValueO.json")


def train_montecarlo():
    p1 = MCAgent(1)
    p2 = MCAgent(2)
    __train_agents(p1, p2, "MCValueX.json", "MCValueO.json")


if __name__ == "__main__":

    REWARDS = {
        GameStatus.DRAW: 0,
        GameStatus.P1_WIN: 1,
        GameStatus.P2_WIN: -1
    }

    print("Welcome to TicTacToe")
    print("Please select an option...")
    print("1: Play human vs human")
    print("2: Train a MonteCarlo agent")
    print("3: Train a TD(0) agent")
    options = {1: human_v_human, 2: train_montecarlo, 3: train_TD}
    str_in = input()
    success = False
    while success is False:
        try:
            choice = options[int(str_in)]
            success = True
        except (TypeError, KeyError) as e:
            print("Invalid input:", e)
            str_in = input()
    choice()
