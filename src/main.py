from TicTacToe import Board, GameStatus
from MonteCarloAgent import MonteCarloAgent, play_match
import random
import json
from MCPlotter import MCPlotter
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


def monte_v_monte():
    """
    Train two monte carlo AIs against each other.
    Agent 1 is always X, Agent 2 is always O.
    Init random Value Fn for both agents
    :param
    :return:
    """

    # learning params
    epsilon = 0.01  # % chance of a random move. updated periodically after playing some number of games
    gamma = 0.90  # decay rate for rewards
    alpha = 0.02  # learning rate
    games_played = 0


    p1 = MonteCarloAgent(1)
    p2 = MonteCarloAgent(2)
    mcplot = MCPlotter(p1, p2)

    startermove = 1

    plot = False  # only plot if we've seen all the start states (plotter code breaks otherwise)
    while games_played < 1_000_000:

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
        p1.update_value(game_log, REWARDS[outcome], gamma=gamma, alpha=alpha)
        p2.update_value(game_log, REWARDS[outcome], gamma=gamma, alpha=alpha)

        # logging and plotting
        if plot:
            openers = {1: "Corner", 2: "Side", 5: "Centre"}
            mcplot.log_and_plot(outcome, opener=openers[startermove])

        # save value fn
        if games_played % 500 == 0:
            with open(f"MCValue.json", "w") as f:
                print(f"saving value function... ({len(p1.value)} states seen, {games_played} games played)")
                d = {str(board): val for board, val in p1.value.items()}
                json.dump(d, f)


def inspect_value():
    value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\MCValue.json"
    optimal_value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\DPSolver.py"
    board = Board()
    p1 = MonteCarloAgent(1, board)
    p2 = MonteCarloAgent(2, board)

    p1.load_value(value_path)
    p2.load_value(value_path)

    for startermove in [1, 2, 5]:
        outcome, gamelog = play_match(p1, p2, epsilon=0, startermove=startermove)
        for boardstate in gamelog:
            print(f"value={p1.value[boardstate]}")
            print(boardstate)
            print("\n" * 3)

    opt_agent = MonteCarloAgent(1, board)
    try:
        opt_agent.load_value(optimal_value_path)
    except FileNotFoundError:
        print("Couldn't load optimal value function")
    else:
        non_draws_missing = 0  # number of "winning" states not found in p1.value
        wins_missing = 0
        sq_error = 0
        for state in opt_agent.value:
            if state in p1.value:
                sq_error += (p1.value[state]-opt_agent.value[state])**2
            else:
                non_draws_missing += (opt_agent.value[state] != 0)
                wins_missing += (opt_agent.value[state] == 1) or (opt_agent.value[state] == -1)

        rmse = np.sqrt((sq_error/len(p1.value)))
        print(f"Among seen states, value function has average error (RMSE) of {rmse}")
        print(f"Among {len(opt_agent.value)-len(p1.value)} unseen states, {non_draws_missing} do not result in a draw")



if __name__ == "__main__":

    REWARDS = {
        GameStatus.DRAW: 0,
        GameStatus.P1_WIN: 1,
        GameStatus.P2_WIN: -1
    }

    print("Welcome to TicTacToe")
    print("Please select an option...")
    print("1: Human v Human")
    print("2: MonteCarlo v MonteCarlo")
    print("3: Inspect/test value function")
    options = {1: human_v_human, 2: monte_v_monte, 3: inspect_value}
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
