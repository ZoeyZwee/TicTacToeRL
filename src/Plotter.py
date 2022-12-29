"""
For plotting fun stuff with MonteCarlo Agents
"""
import matplotlib.pyplot as plt
from TicTacToe import Board
import numpy as np
from collections import namedtuple

Data = namedtuple("Data", ["data", "colour", "label"])

class Plotter:
    def __init__(self, p1, p2):
        self.plt_init()
        # select plots
        self.plot_openers = True
        self.plot_winrates = False
        self.plot_wins = False
        self.plot_last100 = True
        self.plot_open100 = True

        # objects
        self.p1 = p1
        self.p2 = p2

        # opening moves (used to index into value fn to track values)
        corner = Board()
        corner.play_move(9, 1)
        side = Board()
        side.play_move(6, 1)
        centre = Board()
        centre.play_move(5, 1)
        self.openers = [corner, side, centre]

        outcome_labels = ["Draw", "P1 Win", "P2 Win"]
        open_labels = ["Corner", "Side", "Centre"]
        self.open_labels = open_labels
        outcome_colours = ["Red", "Green", "Blue"]
        open_colours = ["c", "m", "y"]


        # outcome logs
        self.games_played = 0
        self.wins = [0, 0, 0]  # number of draws, p1_wins, p2_wins
        self._last100 = {o: [0] * 100 for o in self.open_labels}  # buffer of last 100 games. organized by opening move

        self.win_log = [Data([], c, l) for c, l in zip(outcome_colours, outcome_labels)]
        self.winrates = [Data([], c, l) for c, l in zip(outcome_colours, outcome_labels)]  # history of winrate
        self.last100wr = [Data([], c, l) for c, l in zip(outcome_colours, outcome_labels)]
        # each opener gets a plot akin to last100wr
        self.open100 = {o: [Data([], c, l) for c, l in zip(outcome_colours, outcome_labels)] for o in open_labels}
        self.opener_value = [Data([], c, l) for c, l in zip(open_colours, open_labels)]

        self.init_figures()

    def init_figures(self):
        def init_fig(dataset, title, xlabel):
            fig, ax = plt.subplots(figsize=(12,7))
            for data in dataset:
                ax.plot(data.data, color=data.colour, label=data.label)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            return fig, ax

        if self.plot_wins:
            self.winfig, self.winax = init_fig(
                self.win_log,
                "Outcomes",
                "Total Games Played (hundreds)"
            )

        if self.plot_winrates:
            self.wrfig, self.wrax = init_fig(
                self.winrates,
                "Outcome Rates",
                "Total Games Played (hundreds)"
            )
        if self.plot_last100:
            self.last100fig, self.last100ax = init_fig(
                self.last100wr,
                "Outcome Rates (last 100 games)",
                "Total Games Played (hundreds)"
            )
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_geometry("+1200+0")

        if self.plot_openers:
            self.openfig, self.openax = init_fig(
                self.opener_value,
                "Opening Move Values",
                "Total Games Played (hundreds)"
            )
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_geometry("+1200+700")

        if self.plot_open100:
            self.open100fig, self.open100axs = plt.subplots(3, 1, figsize=(12,20))
            for ax, (opener_name, dataset) in zip(self.open100axs, self.open100.items()):
                for data in dataset:
                    ax.plot(data.data, color=data.colour, label=data.label)
                ax.set_title(opener_name)
            self.open100fig.suptitle("Outcome Rates (last 100 games) by opening move")
            self.open100axs[2].set_xlabel("Total Games Played (hundreds)")
            self.open100axs[0].legend()
            self.open100fig.canvas.draw()
            self.open100fig.canvas.flush_events()
            thismanager = plt.get_current_fig_manager()
            thismanager.window.wm_geometry("+0+0")

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
        opener_winrates_100 = {}  # winrate over last 100 for each opener
        collected_winrates_100 = np.zeros((3))
        for open_str in self.open_labels:
            outcomes_last_100 = np.array(self._last100[open_str])
            rates_last_100 = [np.sum(outcomes_last_100==i)/100 for i in range(3)]
            opener_winrates_100[open_str] = rates_last_100
            collected_winrates_100 += rates_last_100
        winrates_100 = collected_winrates_100/3
        winrates = [score / self.games_played for score in self.wins] if self.plot_winrates else None
        return winrates, winrates_100, opener_winrates_100

    def log_and_plot(self, outcome, opener):
        """
        record the outcome of a game. update trackers
        """
        self.games_played += 1
        self.wins[outcome] += 1
        self._last100[opener].pop(0)
        self._last100[opener].append(outcome)
        # update logs every 100 games
        if self.games_played % 100 != 0:
            return

        # calculate winrates
        wr, wr100, opener_wr100 = self.current_winrates()

        for i in range(3):  # i=0 means draw, i=1 means p1 win, i=2 means p2 win
            if self.plot_winrates:
                self.winrates[i].data.append(wr[i])
            if self.plot_wins:
                self.win_log[i].data.append(self.wins[i])
            if self.plot_last100:
                self.last100wr[i].data.append(wr100[i])
            if self.plot_last100:
                for open_str in self.open_labels:
                    self.open100[open_str][i].data.append(opener_wr100[open_str][i])
            if self.plot_openers:
                move = self.openers[i]
                value = self.p1.value[move]
                self.opener_value[i].data.append(value)

        if self.games_played % 1000 == 0:
            self.make_plots()

    def make_plots(self):
        def plot(fig, ax, dataset):
            for data in dataset:
                ax.plot(data.data, color=data.colour, label=data.label)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if self.plot_wins:
            plot(self.winfig, self.winax, self.win_log)
            self.winfig.savefig("wins.png")

        if self.plot_winrates:
            plot(self.wrfig, self.wrax, self.winrates)
            self.wrfig.savefig("outcomes.png")

        if self.plot_last100:
            plot(self.last100fig, self.last100ax, self.last100wr)
            self.last100fig.savefig("outcomes100.png")

        if self.plot_openers:
            plot(self.openfig, self.openax, self.opener_value)
            self.last100fig.savefig("openers.png")

        if self.plot_open100:
            # self.open100 is structured like {opener_name: [Data*3]}
            for ax, dataset in zip(self.open100axs, self.open100.values()):
                plot(self.open100fig, ax, dataset)
            self.open100fig.savefig("open100.png")

