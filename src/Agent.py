"""
An agent to learn and play tic tac toe via Monte Carlo Control.
"""
from TicTacToe import Board, GameStatus
import random
import numpy as np
import json
from abc import ABC, abstractmethod


def play_match(agent1, agent2, startermove=None):
    """
    Get two Agents to play against each other
    :param agent1: agent playing as X
    :param agent2: agent playing as O
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
    if startermove is not None:  # play chosen starter move
        status = agent1.play_move(startermove)
    else:
        status = agent1.play_policy_move()
    game_log.append(board.copy())
    player = agent2

    # play out match
    while status == GameStatus.RUNNING:
        status = player.play_policy_move()
        game_log.append(board.copy())
        player = agent2 if (player == agent1) else agent1

    return status, game_log


class _AgentABC(ABC):
    def __init__(self, player_id, alpha, gamma, board=None):
        """
        init agent. child classes specify how to update and interpret value function
        :param player_id: 1 for X, 2 for O
        :param alpha: learning rate
        :param gamma: decay rate for future rewards
        :param board: board the agent is playing on
        """
        self.alpha = alpha
        self.gamma = gamma
        self.game = Board() if board is None else board
        # keys to self.value are Boards, values are floats.
        self.value = {}  # lookup table for value fn. >0 means good for p1. <0 means good for p2
        self.player_id = player_id  # 1 if X, 2 if O

    def load_value(self, rpath):
        """
        Load a value function from a .json file at location rpath
        :param rpath: path to .json file
        :return: self
        """

        with open(rpath, 'r') as f:
            data = json.load(f)
        self.value = {Board.from_string(k): v for k, v in data.items()}

        return self

    def new_game(self, board=None):
        self.game = Board() if board is None else board

    def play_move(self, move):
        """
        Play a move and return GameStatus of game after move.
        :param move: position of move to be played (1-9)
        :return: GameStatus of game after playing move
        """
        return self.game.play_move(move, self.player_id)

    @abstractmethod
    def get_best_move(self):
        """
        Determine the "best" move according to the agent's value fn
        """
        pass

    @abstractmethod
    def play_policy_move(self):
        """
        Play a move according to the agent's policy.
        """
        pass

    @abstractmethod
    def get_value(self, board):
        """
        Look up the value of a given afterstate
        """
        pass

    @abstractmethod
    def train(self, afterstates, reward):
        """
        Update value of afterstates, given end-game reward
        """
        pass


class TreeSearchAgent(_AgentABC):

    def search(self):
        """
        3-ply minimax search
        """
        raise NotImplementedError

    def get_best_move(self):
        raise NotImplementedError

    def play_policy_move(self):
        raise NotImplementedError

    def get_value(self, board):
        raise NotImplementedError

    def train(self, afterstates, reward):
        raise NotImplementedError


class TDLeafAgent(TreeSearchAgent):
    pass


class EpsilonAgent(_AgentABC):
    """
    Epsilon greedy agent.
    Value function is randomly initialized.
    Load a pre-trained value function with load_value().
    """

    def __init__(self, player_id, alpha=0.1, gamma=0.9, epsilon=0.1, board=None):
        self.epsilon = epsilon  # exploration rate
        super().__init__(player_id, alpha, gamma, board=board)

    def get_value(self, board):
        return self.value[board]

    def play_policy_move(self):
        """
        get agent to play the best move with epsilon chance of playing a random move
        :return: GAME_STATUS of game after executing move
        """

        if random.random() < self.epsilon:
            legals = self.game.get_legals()
            return self.play_move(np.random.choice(legals))
        else:
            return self.play_move(self.get_best_move())

    def get_best_move(self):
        """
        Simulate all legal moves, decide which is best for us.
        Positive value is good for X
        Negative value is good for O
        :return: # from 1-9 indicating the best move
        """
        best_value = None
        best_move = None

        def is_new_best(val):
            # p1 wants to maximize value, p2 wants to minimize
            return val > best_value if self.player_id == 1 else val < best_value

        for move in self.game.get_legals():
            # simulate move to get value of resulting state
            afterstate = self.game.sim_move(move, self.player_id)

            # add state to value fn on first visit
            if afterstate not in self.value.keys():
                self.value[afterstate] = random.uniform(-1, 1)

            move_value = self.value[afterstate]

            # update best_value, best_move
            if best_value is None:
                best_value = move_value
                best_move = move
            elif is_new_best(move_value):
                best_value = move_value
                best_move = move

        return best_move

    def train(self, afterstates, reward):
        raise NotImplementedError("base class EpsilonAgent doesn't know how to train()")


class TDAgent(EpsilonAgent):
    """
    An agent that learns w/ TD(0) and epsilon-greedy policy.
    """
    def train(self, afterstates, reward):
        """
        Update the value function using TD(0) update for value fn.
        Doesn't need to be online since states are never re-visited during an episode

        :param afterstates: list of Boards, corresponding to all the game states AFTER our agent has played
        :param reward: +1 for p1 win, -1 for p2 win, 0 for draw
        :return: None
        """
        # update value of last afterstate.
        # if we played the final move, then the afterstate is a terminal state,
        #   so we set the afterstate value to be *exactly* the transition reward
        if afterstates[-1].running_state() != GameStatus.RUNNING:
            self.value[afterstates[-1]] = reward
        else:
            # Future rewards are zero, so move value towards actual reward
            expected_reward = self.value[afterstates[-1]]
            self.value[afterstates[-1]] = expected_reward + self.alpha * (reward - expected_reward)

        # remaining updates bootstrap toward the next afterstate
        next_afterstate = afterstates[-1]  # start at the end, iterate backwards.
        for afterstate in afterstates[-1::-1]:
            if afterstate not in self.value.keys():
                # sometimes states don't make it into our value function during play (usually due to random moves)
                self.value[afterstate] = self.gamma*self.value[next_afterstate]
            else:
                # transition reward is zero, so TD-target is decayed estimate of future rewards
                future_value = self.gamma * self.value[next_afterstate]
                current_value = self.value[afterstate]
                self.value[afterstate] = current_value + self.alpha * (future_value - current_value)

            next_afterstate = afterstate


class MCAgent(EpsilonAgent):
    """
    An agent that learns w/ MonteCarlo Control
    """
    def train(self, afterstates, reward):
        """
        Update the value function using the bellman equation
        :param afterstates: list of Boards, corresponding to a single game
        :param reward: +1 for p1 win, -1 for p2 win, 0 for draw
        :return: None
        """
        for decay_steps, afterstate in enumerate(reversed(afterstates)):
            rtn = reward * self.gamma**decay_steps  # decayed future reward (i.e. return)
            if afterstate not in self.value.keys():
                self.value[afterstate] = rtn
            # incorporate trajectory into average
            self.value[afterstate] = self.value[afterstate] + self.alpha*(rtn-self.value[afterstate])


if __name__ == "__main__":
    """
    Examine and test the value function specified by value_path.
    Compare that value function with an optimal value function
    """
    x_value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\TDValueX.json"
    o_value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\TDValueO.json"
    optimal_value_path = "D:\Programming\GithubRepos\TicTacToeRL\src\DPValue.json"
    p1 = EpsilonAgent(1, epsilon=0).load_value(x_value_path)
    p2 = EpsilonAgent(2, epsilon=0).load_value(o_value_path)
    jointvalue = p1.value | p2.value  # merge the two value functions, for getting statistics
    p1.value = jointvalue
    p2.value = jointvalue

    for startermove in [1, 2, 5]:
        outcome, gamelog = play_match(p1, p2, epsilon=0, startermove=startermove)
        for boardstate in gamelog:
            print(f"value={p1.value[boardstate]}")
            print(boardstate)
            print("\n" * 2)

    opt_agent_X = EpsilonAgent(1, epsilon=0)
    opt_agent_O = EpsilonAgent(2, epsilon=0)  # create an optimal "O" player

    opt_agent_X.load_value(optimal_value_path)
    opt_agent_O.value = opt_agent_X.value

    terminals_missing = [0,0,0]  # number of [draws, X-win, O-win] states not found in p1.value
    sq_error = 0
    for state in opt_agent_X.value:
        if state in p1.value:
            sq_error += (p1.value[state] - opt_agent_X.value[state]) ** 2
        else:
            if (r := state.running_state()) in [0,1,2]:
                terminals_missing[r] += 1

    rmse = np.sqrt((sq_error / len(p1.value)))
    print(f"Among seen states, value function has average error (RMSE) of {rmse:.3f}")
    print(f"There are {len(opt_agent_X.value) - len(p1.value)} unseen states")
    print(f"Of the unseen states, {sum(terminals_missing)} are terminal states.")
    print(f"{terminals_missing[0]} are drawn states, {terminals_missing[1]} are X-wins, {terminals_missing[2]} are O-wins")
    print(f"For reference, there exist 135 winning states (states where a player has won)")

    # play out a game w/ learned agent vs optimal
    p_vs_opt = []
    opt_vs_p = []
    strmap = lambda a: "X win" if a==1 else "O win" if a==2 else "Draw"
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

    # random agents
    rand_agent_X = EpsilonAgent(1, epsilon=1)
    rand_agent_O = EpsilonAgent(2, epsilon=1)
    for i in range(1000):

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
