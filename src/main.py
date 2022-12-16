from TicTacToe import Board


def get_move(player):
    x = ""
    ins = [1,2,3,4,5,6,7,8,9]
    while x not in ins:
        try:
            x = int(input(f"Player {player}: "))
        except Exception:
            pass
    return x


if __name__ == "__main__":
    tt = Board()
    p1 = True
    print("A new game of tic tac toe wow!!!!")
    print("Valid inputs are 1-9")
    for i in range(9):
        player = 1 if p1 else 2
        winner = tt.playmove(get_move(player), player)
        p1 = not p1
        print(tt)
        if winner != 0:
            print(f"Player {winner} has won!!!")
            break
