"""
An agent to learn and play tic tac toe via Monte Carlo Control.
"""
from TicTacToe import Board, GameStatus
import random
import numpy as np
import json


def play_match(agent1, agent2, epsilon=0, startermove=None):
    """
    Get two MonteCarloAgents to play against each other
    :param agent1: agent playing as X
    :param agent2: agent playing as O
    :param epsilon: chance for either agent to make a random move
    :param startermove: optional starter move (1-9)
    :return: GameStatus: status (result of game), [Board]: game_log (list of board states seen in game)
    """
    # init
    game_log = []
    player = agent1  # player who is next to move
    board = Board()
    agent1.new_game(board)
    agent2.new_game(board)

    # optional fixed first move
    if startermove is not None: # play chosen starter move
        status = agent1.play_move(startermove)
    else:
        status = agent1.epsilon_best_move(1)
    game_log.append(board.copy())
    player = agent2

    # play out match
    while status == GameStatus.RUNNING:
        status = player.epsilon_best_move(epsilon)
        game_log.append(board.copy())
        player = agent2 if (player == agent1) else agent1

    return status, game_log


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

    def load_value(self, rpath):
        """
        Load a value function from a .json file at location rpath
        :param rpath:
        :return:
        """

        with open(rpath, 'r') as f:
            data = json.load(f)
        self.value = {Board.from_string(k): v for k, v in data.items()}

    def new_game(self, board=None):
        self.game = Board() if board is None else board

    def play_move(self, move):
        """
        Play a move and return who won (1 for p1, 2 for p2, 0 for game still on, -1 for draw
        :param move: position of move to be played (1-9)
        :return: 1 if p1 won. 2 if p2 won. 0 for game still going. -1 for draw.
        """
        return self.game.play_move(move, self.player_id)

    def epsilon_best_move(self, epsilon):
        """
        get agent to play the best move with epsilon chance of playing a random move
        :param epsilon: chance we play a random move
        :return: GAME_STATUS of game after executing move
        """

        if random.random() < epsilon:
            return self.__play_random_move()
        else:
            return self.play_move(self.get_best_move())

    def __play_random_move(self):
        """
        Play a random move. For external use, call epsilon_best_move(epsilon=1)
        :return: GAME_STATUS of game after executing move
        """
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
                self.value[next_state] = random.uniform(-1, 1)

            move_value = self.value[next_state]

            # update best_value, best_move
            if best_value is None:
                best_value = move_value
                best_move = move
            elif compare(move_value):
                best_value = move_value
                best_move = move

        return best_move

    def update_value(self, states_visited, reward, gamma, alpha):
        """
        Update the value function using the bellman equation
        :param states_visited: list of Boards, corresponding to a single game
        :param reward: +1 for p1 win, -1 for p2 win, 0 for draw
        :param gamma: decay rate for rewards
        :param alpha: learning rate
        :return: None
        """
        for i, state in enumerate(states_visited):
            decay_steps = len(states_visited)-i-1  # on last move we have len(game)==i, and we want 0 decay
            rtn = reward * gamma**decay_steps  # decayed future reward (i.e. return)
            if state not in self.value.keys():
                # sometimes moves by the other player don't get initialized in our value fn
                self.value[state] = random.uniform(-1, 1)
            # incorporate trajectory into average
            self.value[state] = self.value[state] + alpha*(rtn-self.value[state])


if __name__ == "__main__":
    """
    Examine and test the value function specified by value_path.
    Compare that value function with an optimal value function
    """
    value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\MCValue.json"
    optimal_value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\DPValue.json"
    p1 = MonteCarloAgent(1)
    p2 = MonteCarloAgent(2)

    p1.load_value(value_path)
    p2.load_value(value_path)

    for startermove in [1, 2, 5]:
        outcome, gamelog = play_match(p1, p2, epsilon=0, startermove=startermove)
        for boardstate in gamelog:
            print(f"value={p1.value[boardstate]}")
            print(boardstate)
            print("\n" * 2)

    opt_agent_X = MonteCarloAgent(1)
    try:
        opt_agent_X.load_value(optimal_value_path)
    except FileNotFoundError:
        print("Couldn't load optimal value function - file not found")
    else:
        non_draws_missing = 0  # number of "winning" states not found in p1.value
        wins_missing = 0
        sq_error = 0
        for state in opt_agent_X.value:
            if state in p1.value:
                sq_error += (p1.value[state] - opt_agent_X.value[state]) ** 2
            else:
                non_draws_missing += (opt_agent_X.value[state] != 0)
                wins_missing += (opt_agent_X.value[state] == 1) or (opt_agent_X.value[state] == -1)

        rmse = np.sqrt((sq_error / len(p1.value)))
        print(f"Among seen states, value function has average error (RMSE) of {rmse:.3f}")
        print(f"Among {len(opt_agent_X.value) - len(p1.value)} unseen states, {non_draws_missing} do not result in a draw")
        print(f"Among {len(opt_agent_X.value) - len(p1.value)} unseen states, {non_draws_missing} are winning states")
        print(f"For reference, there are 135 winning states (states where a player has won)")

    # play out a game w/ learned agent vs optimal
    opt_agent_O = MonteCarloAgent(2)  # create an optimal "O" player
    opt_agent_O.value = opt_agent_X.value

    p_vs_opt = []
    opt_vs_p = []
    strmap = lambda o: "X win" if o==1 else "O win" if o==2 else "Draw"
    for startermove in [1,5,2]:
        outcome, _ = play_match(p1, opt_agent_O, startermove=startermove)
        p_vs_opt.append(strmap(outcome))
        outcome, _ = play_match(opt_agent_X, p2, startermove=startermove)
        opt_vs_p.append(strmap(outcome))
    print()
    print("Learned Agent as X, Optimal Agent as O")
    print("Corner Middle Side")
    print(p_vs_opt)

    print()
    print("Optimal Agent as X, Learned Agent as O")
    print("Corner Middle Side")
    print(opt_vs_p)

    print()
    print("playing out 1000 games vs random opponent...")
    # p1 and optimal agent both play out 1000 games vs random opponent, as both X and O
    p_scores_X = np.array([0,0,0])
    opt_scores_X = np.array([0,0,0])
    p_scores_O = np.array([0,0,0])
    opt_scores_O = np.array([0,0,0])


    for i in range(1000):

        # init blank agents (init every time since policy is stationary if states have been seen before)
        rand_agent_X = MonteCarloAgent(1)
        rand_agent_O = MonteCarloAgent(2)

        # player, X
        outcome, _ = play_match(p1, rand_agent_O)
        p_scores_X[outcome] += 1

        # optimal, X
        outcome, _ = play_match(opt_agent_X, rand_agent_O)
        opt_scores_X[outcome] += 1

        # player, O
        outcome, _ = play_match(rand_agent_X, p2)
        p_scores_O[outcome] += 1

        # optimal, O
        outcome, _ = play_match(rand_agent_X, opt_agent_O)
        opt_scores_O[outcome] += 1

    p_rates_X = (p_scores_X/1000)[[1, 0, 2]]
    p_rates_O = (p_scores_O/1000)[[2, 0, 1]]
    opt_rates_X = (opt_scores_X/1000)[[1, 0, 2]]
    opt_rates_O = (opt_scores_O/1000)[[2, 0, 1]]
    print(f"Learned Value fn outcome rates vs random opponent...\n"
          f"     W    D    L\n"
          f"X: {p_rates_X}\n"
          f"O: {p_rates_O}\n\n"
          f"Optimal Value fn outcome rates vs random opponent...\n"
          f"     W    D    L\n"
          f"X: {opt_rates_X}\n"
          f"O: {opt_rates_O}\n")
