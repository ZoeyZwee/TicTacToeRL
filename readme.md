# Tic Tac Toe RL

A fun little project where I try to build agents to play Tic Tac Toe with various methods.   
My hope is to reach the optimal policy for every method used.

Progress so far:
 - Monte Carlo Control: Was not able to consistently draw vs an optimal opponent after 500k. Sometimes lost to a fully random opponent as both X and O
 - TD(0): Was able to consistently draw vs an optimal opponent after 500k training games. Loses very occasionally to a fully random opponent as O, never loses as X. Wins vs random at roughly the same rate as a perfect player.