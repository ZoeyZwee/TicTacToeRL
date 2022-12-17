"""
For plotting fun stuff with MonteCarlo Agents
"""
import matplotlib.pyplot as plt
from TicTacToe import Board


class MCPlotter:
    def __init__(self, p1, p2):
        self.plt_init()
        # select plots
        self.plot_openers = True
        self.plot_winrates = True
        self.plot_wins = True

        # objects
        self.p1 = p1
        self.p2 = p2

        # outcome logs
        self.games_played = 0
        self.wins = [0, 0, 0]  # number of draws, p1_wins, p2_wins
        self.win_log = [[], [], []]
        self.winrates = [[], [], []]  # history of winrate

        # opening moves (we want to track value of these)
        self.corner = Board()
        self.corner.play_move(9, 1)
        self.side = Board()
        self.side.play_move(6, 1)
        self.centre = Board()
        self.centre.play_move(5, 1)

        # opener logs
        self.openers = [self.corner, self.side, self.centre]
        self.opener_value = {self.corner: [], self.side: [], self.centre: []}
        self.opener_visits = {self.corner: [], self.side: [], self.centre: []}

        if self.plot_wins:
            self.winplot, self.winax = plt.subplots()
            self.winax.set_title('Outcomes')
            # dummy plots for legend
            self.winax.plot(self.win_log[0], color="red", label="Draws")
            self.winax.plot(self.win_log[1], color="green", label="P1 Wins")
            self.winax.plot(self.win_log[2], color="blue", label="P2 Wins")
            self.winax.legend()
        if self.plot_winrates:
            self.wrplot, self.wrax = plt.subplots()
            # dummy plots for legend
            self.wrax.plot(self.winrates[0], color="red", label="Draw rate")
            self.wrax.plot(self.winrates[1], color="green", label="P1 Win rate")
            self.wrax.plot(self.winrates[2], color="blue", label="P2 Win rate")
            self.wrax.set_title('Outcome Rates')
            self.wrax.legend()
        if self.plot_openers:
            self.openplot, self.openax = plt.subplots(2)
            # dummy plots for legend
            self.openax[0].plot(self.opener_value[self.corner], color="red", label="Corner")
            self.openax[0].plot(self.opener_value[self.centre], color="green", label="Centre")
            self.openax[0].plot(self.opener_value[self.side], color="blue", label="Side")
            self.openax[1].plot(self.opener_visits[self.corner], color="red", label="Corner")
            self.openax[1].plot(self.opener_visits[self.centre], color="green", label="Centre")
            self.openax[1].plot(self.opener_visits[self.side], color="blue", label="Side")
            self.openax[0].set_title('Opener Value')
            self.openax[0].legend()
            self.openax[1].set_title('Opener Visits')
            self.openax[1].legend()

    def plt_init(self):
        # make pyplot print to external backend
        candidates = ["macosx", "qt5agg", "gtk3agg", "tkagg", "wxagg"]
        for candidate in candidates:
            try:
                plt.switch_backend(candidate)
                print('Using backend: ' + candidate)
                break
            except (ImportError, ModuleNotFoundError):
                pass

        # turn on interactive pyplot
        plt.ion()

    def current_winrates(self):
        return [score / self.games_played for score in self.wins]

    def log_and_plot(self, outcome):
        """
        record the outcome of a game. update trackers
        """
        self.games_played += 1
        self.wins[outcome] += 1

        # update logs and plots every 100 games
        if self.games_played % 100 == 0:
            # wins and winrates
            wr = self.current_winrates()
            for i in range(3):
                self.winrates[i].append(wr[i])
                self.win_log[i].append(self.wins[i])
            # openers
            for move in self.openers:
                value = self.p1.value[move].value
                visits = self.p1.value[move].visits
                self.opener_value[move].append(value)
                self.opener_visits[move].append(visits)

            self.make_plots()

    def make_plots(self):
        if self.plot_wins:
            fig, ax = (self.winplot, self.winax)
            ax.plot(self.win_log[0], color="red", label="Draws")
            ax.plot(self.win_log[1], color="green", label="P1 Wins")
            ax.plot(self.win_log[2], color="blue", label="P2 Wins")
            ax.set_title('Outcomes')
            fig.canvas.draw()
            fig.canvas.flush_events()

        if self.plot_winrates:
            fig, ax = (self.wrplot, self.wrax)
            ax.plot(self.winrates[0], color="red", label="Draw rate")
            ax.plot(self.winrates[1], color="green", label="P1 Win rate")
            ax.plot(self.winrates[2], color="blue", label="P2 Win rate")

            fig.canvas.draw()
            fig.canvas.flush_events()

        if self.plot_openers:
            fig, axs = (self.openplot, self.openax)
            axs[0].plot(self.opener_value[self.corner], color="red", label="Corner")
            axs[0].plot(self.opener_value[self.centre], color="green", label="Centre")
            axs[0].plot(self.opener_value[self.side], color="blue", label="Side")
            axs[0].set_title('Opener Values')

            axs[1].plot(self.opener_visits[self.corner], color="red", label="Corner")
            axs[1].plot(self.opener_visits[self.centre], color="green", label="Centre")
            axs[1].plot(self.opener_visits[self.side], color="blue", label="Side")
            axs[1].set_title('Opener Visits')
            fig.canvas.draw()
            fig.canvas.flush_events()
