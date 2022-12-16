"""
A file to play Tic Tac Toe via the console.
"""

import numpy as np

class Board:
    """
    Smart game board. Crucially, game boards are "equivalent" under rotational symmetry
    """
    def __init__(self, board=np.zeros([3,3])):
        self.board = board  # 0=free. 1=player1=X. 2=player2=O

    def __repr__(self):
        XO = lambda val: " " if val==0 else "X" if val==1 else "O"
        a = [XO(x) for x in np.nditer(self.board)]
        str = (
            f"{a[0]} | {a[1]} | {a[2]}\n"
            f"{'-'*9}\n"
            f"{a[3]} | {a[4]} | {a[5]}\n"
            f"{'-' * 9}\n"
            f"{a[6]} | {a[7]} | {a[8]}\n"
        )
        return str

    def __idx_to_coords(self, idx):
        """
        given a "numpad-style" board position, return the coords into a 3x3 ndarray
        """
        row = (10-idx-1) // 3
        col = (idx-1) % 3
        return row, col

    def __coords_to_idx(self, row, col):
        return 1 + (3 * (2-row)) + col

    def playmove(self, idx, player):
        """
        player `player` played on square `idx`
        :param idx: # from 1-9 with ordering same as numpad
        :param player: 1 for p1, 2 for p2
        """
        row, col = self.__idx_to_coords(idx)
        self.board[row][col] = player
        return self.checkWin()

    def checkWin(self):
        """
        returns id of winning player (1 or 2) if someone has won
        returns 0 if nobody has won
        """
        # check rows/cols
        for i in range(3):
            row = self.board[i, :]
            col = self.board[:, i]
            for ax in [row, col]:
                if ax[0] != 0 and np.all(ax == ax[0]):
                    return ax[0]

        # check diagonals
        d1 = self.board[(0,0), (1,1), (2,2)]
        d2 = self.board[(0,2), (1,1), (2,0)]
        for ax in [d1, d2]:
            if ax[0] != 0 and np.all(ax == ax[0]):
                return ax[0]

        # no winner. return 0
        return 0

    def __eq__(self, other):
        r1 = np.rot90(self.board)
        r2 = np.rot90(r1)
        r3 = np.rot90(r2)
        return self.board == r1 or self.board == r2 or self.board == r3
