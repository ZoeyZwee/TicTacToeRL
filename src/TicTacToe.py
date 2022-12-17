"""
A file to play Tic Tac Toe :)
"""

import numpy as np
from enum import Enum


class GameStatus:
    DRAW = 0
    P1_WIN = 1
    P2_WIN = 2
    RUNNING = 3


def idx_to_coords(pos):
    """
    given a "numpad-style" board position, return the coords into a 3x3 ndarray
    """
    row = (10 - pos - 1) // 3
    col = (pos - 1) % 3
    return row, col


def coords_to_idx(row, col):
    return 1 + (3 * (2 - row)) + col


class Board:
    """
    Smart game board for TicTacToe
    """
    def __init__(self, board=np.zeros([3,3], dtype=int)):
        """
        :param board: 3x3 numpy array, where 0=free, 1=X, 2=O
        """
        self.board = board.copy()  # 0=free. 1=player1=X. 2=player2=O

    def play_move(self, pos, player):
        """
        player `player` played on square `pos`
        :param pos: # from 1-9 with ordering same as numpad
        :param player: 1 for p1, 2 for p2
        """
        row, col = idx_to_coords(pos)
        self.board[row][col] = player
        return self.running_state()

    def sim_move(self, pos, player):
        """
        Simulate a move by creating a copy of self and playing the move on that.
        Return the copy.
        :param pos: # from 1-9, same ordering as numpad
        :param player: 1 for p1 (X), 2 for p2 (O)
        :return: An instance of Board where the move has been played
        """
        sim_board = self.copy()
        sim_board.play_move(pos, player)
        return sim_board

    def running_state(self):
        """
        returns id of winning player (1 or 2) if someone has won
        returns RunningState.DRAW or RunningState.RUNNING if nobody has won
        """
        # check rows/cols
        for i in range(3):
            row = self.board[i, :]
            col = self.board[:, i]
            for ax in [row, col]:
                if ax[0] != 0 and np.all(ax == ax[0]):
                    return ax[0]  # return ID of winning player

        # check diagonals
        d1 = self.board[[0, 1, 2], [0, 1, 2]]  # (0,0), (1,1), (2,2)
        d2 = self.board[[0, 1, 2], [2, 1, 0]]  # (0,2), (1,1), (2,0)
        for ax in [d1, d2]:
            if ax[0] != 0 and np.all(ax == ax[0]):
                return ax[0]  # return ID of winning player

        # check if board is full
        if len(self.get_legals()) == 0:
            return GameStatus.DRAW

        # no winner. return 0
        return GameStatus.RUNNING

    def equivs(self):
        """
        Generate all boards which are equivalent to board
        :return: list of board states after apply rotations and reflections
        """
        r0 = self.board
        r1 = np.rot90(r0)
        r2 = np.rot90(r1)
        r3 = np.rot90(r2)
        href = self.board[:, [2, 1, 0]]
        vref = self.board[[2, 1, 0], :]
        return [r0, r1, r2, r3, vref, href]

    def get_legals(self):
        """
        :return: list of positions (1,2,3..9) indicating legal moves
        """
        return np.nonzero(self.get_flat() == 0)[0] + 1

    def get_flat(self):
        """
        :return: returns a "flat" representation of the board, where index 0 is position 1
        and index 8 is position 9
        """
        return np.concatenate((self.board[2], self.board[1], self.board[0]))

    def copy(self):
        return Board(self.board)

    def __repr__(self):
        XO = [" ", "X", "O"]
        a = [XO[x] for x in np.nditer(self.board)]
        str = (
            f"{a[0]} | {a[1]} | {a[2]}\n"
            f"{'-'*9}\n"
            f"{a[3]} | {a[4]} | {a[5]}\n"
            f"{'-' * 9}\n"
            f"{a[6]} | {a[7]} | {a[8]}"
        )
        return str

    def __hash__(self):
        """
        find the "minimal" transformation of the current board, then hash the string representation
        of that board. Minimal is defined using (<=) for strings.
        transformations are rotation and reflection
        :return: hashed "minimal" transformation
        """
        return min([hash(a.__str__()) for a in self.equivs()])

    def __eq__(self, other):
        """
        check equality, up to rotational and reflection symmetry
        :param other: board to compare against
        :return: true if other is equal to self.board, or a reflection/rotation of self.board
        """
        # for each transformation, check if they are equal to other
        # return true if any of those checks returned true
        return np.any([np.all(other == a) for a in self.equivs()])

    def __len__(self):
        return np.count_nonzero(self.board)


if __name__ == "__main__":
    test1 = Board()
    test2 = Board()
    test1.play_move(1,1)
    test2.play_move(3,1)

    print(np.any([np.all(test2.board == a) for a in test1.equivs()]))

