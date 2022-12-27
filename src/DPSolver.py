"""
A class to find the optimal value function of Tic Tac Toe using dynamic programming
"""

from TicTacToe import Board, GameStatus
import json
## generate states and transitions
# start from empty board
# create all board states after 1 move
# create dict linking each board state to all resulting board states
# repeat until full network of board states created, categorizing board states by how many moves have been played

## solve value function
# start at 9-move boards
# assign value to 9-move boards
# look at 8-move boards
# if 8-move board has NO afterstates, it must be a terminal board
#  - check who won, assign value
# else its value is (decayed) min/max of afterstate values
# repeat until all boards have assigned value

def reward(outcome):
    match outcome:
        case GameStatus.DRAW:
            return 0
        case GameStatus.P1_WIN:
            return 1
        case GameStatus.P2_WIN:
            return -1


def optimal_value_fn():
    player = 1  # 1 is X, 2 is O

    # generate all board states (with transitions)
    states = []
    b = Board()
    states.append({b: [b.sim_move(move, player) for move in b.get_legals()]})
    for i in range(1, 10):
        states.append({})
        player = (i % 2) + 1
        for s, afterstates in states[i - 1].items():
            if not afterstates:
                continue
            for a in afterstates:
                # yes we waste computations, no it does not matter
                if a.running_state() == GameStatus.RUNNING:
                    states[i][a] = [a.sim_move(move, player) for move in a.get_legals()]
                else:
                    states[i][a] = []

    # solve value function
    value = {}
    for i in range(9, -1, -1):
        player = (i % 2) + 1
        for state in states[i]:
            if not states[i][state]:
                # state is terminal
                value[state] = reward(state.running_state())
            else:
                # state is non-terminal
                minmax = max if player == 1 else min
                value[state] = 0.9 * minmax([value[afterstate] for afterstate in states[i][state]])

    return value


if __name__ == "__main__":
    v = optimal_value_fn()
    with open(f"DPValue.json", "w") as f:
        d = {str(board): val for board, val in v.items()}
        json.dump(d, f)
