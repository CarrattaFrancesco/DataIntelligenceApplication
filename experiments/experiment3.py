from environment import Environment
from tqdm import tqdm
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import numpy as np
import matplotlib.pyplot as plt 

class Experiment3:

    def __init__(self):
        self.env = Environment()
        n_arms = 13
        self.ucb1_learner = UCB1(n_arms = n_arms)
        self.ts_learner = TS_Learner(n_arms = n_arms)

        self.T = 365
        self.opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
        self.opt_price = 6.321089806558111
        self.regret_ucb = []
        self.regret_ts = []

        # Use this for testing, the price learned is 7, the optimal one
        # bids = opt_bids 
        self.bids = [5.0, 5.0, 5.0]
        self.prices = np.linspace(3.0, 15.0, n_arms)

        self.price_ev_per_day_ucb = []
        self.price_ev_per_day_ts = []

    def run(self):
        for day in range(self.T):
            #UCB1 learner
            price_idx = self.ucb1_learner.pull_arm()
            price = self.prices[price_idx]
            reward = sum(self.env.round(self.bids, price)) #treat as aggregate data
            self.ucb1_learner.update(price_idx, reward)
            self.regret_ucb.append(sum(self.env.round(self.opt_bids, self.opt_price, noise= False)) - reward)
            self.price_ev_per_day_ucb.append(price)

            #TS learner
            price_idx = self.ts_learner.pull_arm()
            price = self.prices[price_idx]
            reward = sum(self.env.round(self.bids, price)) #treat as aggregate data
            self.ts_learner.update(price_idx, reward)
            self.regret_ts.append(sum(self.env.round(self.opt_bids, self.opt_price, noise= False)) - reward)
            self.price_ev_per_day_ts.append(price)

    def showRegret(self):
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(self.regret_ucb , 'r')
        plt.plot(self.regret_ts, "b")
        plt.legend(["UCB", "TS"])
        plt.show()

    def showPriceEvolution(self):
        plt.figure(1)
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.plot(self.price_ev_per_day_ucb , 'r')
        plt.plot(self.price_ev_per_day_ts , 'b')
        plt.grid()
        plt.yticks(self.prices)
        plt.hlines(self.opt_price, 0, 365, 'g', linestyles='dashed', label="optimal price")
        plt.legend(["UCB", "TS", "Optimal Price"])
        plt.title("Price evolution per day")
        plt.show()