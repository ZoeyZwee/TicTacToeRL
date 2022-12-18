from TicTacToe import Board, GameStatus
from TicTacAgent import MonteCarloAgent
import random

from src.MCPlotter import MCPlotter


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

    games_played = 0
    epsilon = 1  # % chance of a random move. epsilon = 1/(k+1) where k is games played

    board = Board()
    p1 = MonteCarloAgent(1, board)
    p2 = MonteCarloAgent(2, board)
    mcplot = MCPlotter(p1, p2)

    startermove = 1

    plot = False  # only plot if we've seen all the start states (plotter code breaks otherwise)
    while True:
        game_log = []  # list of game states from THIS game
        # play a game
        # start with each opener evenly
        match startermove:
            case 1:
                startermove = 2
            case 2:
                startermove = 5
            case 5:
                startermove = 1
                plot = True  # all start states seem from here on out
        p1.play_move(startermove)
        game_log.append(board.copy())
        epsilon_move(p2, epsilon)
        game_log.append(board.copy())
        while True:
            # p1 plays
            status = epsilon_move(p1, epsilon)
            game_log.append(board.copy())
            if status != GameStatus.RUNNING:
                break

            # p2 plays
            status = epsilon_move(p2, epsilon)
            game_log.append(board.copy())
            if status != GameStatus.RUNNING:
                break


        # update value fns
        p1.update_value(game_log, status, gamma=GAMMA, alpha=ALPHA)
        p2.update_value(game_log, status, gamma=GAMMA, alpha=ALPHA)

        # reset board
        board = Board()
        p1.new_game(board)
        p2.new_game(board)

        # logging
        if plot:
            mcplot.log_and_plot(status)

        games_played += 1
        epsilon = 1/(games_played+1)
        # print(status)
        # print("D  P1  P2")
        # print(win_log, f"P1 wr: {win_log[1]/sum(win_log):.3%}, P2 wr: {win_log[2]/sum(win_log):.3%}")


def epsilon_move(agent, epsilon):
    """
    get agent to play a move with epsilon chance of playing a random move
    :param agent: agent which is playing the move
    :param epsilon: chance we play a random move
    :return: RUNNING_STATE of game after executing move
    """

    if random.random() < epsilon:
        return agent.play_random()
    else:
        return agent.play_move(agent.get_best_move())




if __name__ == "__main__":
    # PARAMETERS
    GAMMA = 0.90
    ALPHA = 0.01
    REWARDS = {
        GameStatus.DRAW: 0,
        GameStatus.P1_WIN: 1,
        GameStatus.P2_WIN: -1
    }

    print("Welcome to TicTacToe")
    print("Please select an option...")
    print("1: Human v Human")
    print("2: MonteCarlo v MonteCarlo")

    options = {1: human_v_human, 2: monte_v_monte}
    str_in = input()
    success = False
    options[int(str_in)]()
    while success is False:
        try:
            choice = options[int(str_in)]
            success = True
        except (TypeError, KeyError) as e:
            print("Invalid input:", e)
            str_in = input()
    choice()
    