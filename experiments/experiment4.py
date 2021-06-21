import numpy as np
from environment import Environment
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import matplotlib.pyplot as plt 

class Experiment4:

    def __init__(self):
        self.env = Environment()
        n_arms = 13
        self.ucb1_learner_c0 = UCB1(n_arms = n_arms)
        self.ucb1_learner_c1 = UCB1(n_arms = n_arms)
        self.ucb1_learner_c2 = UCB1(n_arms = n_arms)
        self.ts_learner_c0 = TS_Learner(n_arms = n_arms)
        self.ts_learner_c1 = TS_Learner(n_arms = n_arms)
        self.ts_learner_c2 = TS_Learner(n_arms = n_arms)

        self.T = 365
       
        self.regret_c0_ucb = []
        self.regret_c1_ucb = []
        self.regret_c2_ucb = []

        self.regret_c0_ts = []
        self.regret_c1_ts = []
        self.regret_c2_ts = []

        self.prices = np.linspace(3.0, 15.0, n_arms)

        self.price_ev_per_day_ucb = []
        self.price_ev_per_day_ts = []
    
    def run(self):
        opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
        opt_price = 6.321089806558111
        bids = opt_bids 
        for day in range(self.T):
            #UCB1 learner
            price_c0_idx = self.ucb1_learner_c0.pull_arm()
            price_c0 = self.prices[price_c0_idx]
            price_c1_idx = self.ucb1_learner_c1.pull_arm()
            price_c1 = self.prices[price_c1_idx]
            price_c2_idx = self.ucb1_learner_c2.pull_arm()
            price_c2 = self.prices[price_c2_idx]
            
            reward_per_day = []
            price = [price_c0, price_c1, price_c2]
            for i in range(len(price)):
                p = price[i]
                reward = self.env.round(bids, p)
                reward_per_day.append(reward[i]) 

            self.ucb1_learner_c0.update(price_c0_idx, reward_per_day[0])
            self.ucb1_learner_c1.update(price_c1_idx, reward_per_day[1])
            self.ucb1_learner_c2.update(price_c2_idx, reward_per_day[2])
            
            reward_optimal = self.env.round(self.opt_bids, self.opt_price, noise= False)
            regret_c0 = reward_optimal[0] - reward_per_day[0]
            regret_c1 = reward_optimal[1] - reward_per_day[1]
            regret_c2 = reward_optimal[2] - reward_per_day[2]
            self.regret_c0_ucb.append(regret_c0)
            self.regret_c1_ucb.append(regret_c1)
            self.regret_c2_ucb.append(regret_c2)

            self.price_ev_per_day_ucb.append(price)

            #TS learner
            price_c0_idx = self.ts_learner_c0.pull_arm()
            price_c0 = self.prices[price_c0_idx]
            price_c1_idx = self.ts_learner_c1.pull_arm()
            price_c1 = self.prices[price_c1_idx]
            price_c2_idx = self.ts_learner_c2.pull_arm()
            price_c2 = self.prices[price_c2_idx]
            
            reward_per_day = []
            price = [price_c0, price_c1, price_c2]
            for i in range(len(price)):
                p = price[i]
                reward = self.env.round(bids, p)
                reward_per_day.append(reward[i]) 

            self.ts_learner_c0.update(price_c0_idx, reward_per_day[0])
            self.ts_learner_c1.update(price_c1_idx, reward_per_day[1])
            self.ts_learner_c2.update(price_c2_idx, reward_per_day[2])
            
            reward_optimal = self.env.round(opt_bids, opt_price, noise= False)
            regret_c0 = reward_optimal[0] - reward_per_day[0]
            regret_c1 = reward_optimal[1] - reward_per_day[1]
            regret_c2 = reward_optimal[2] - reward_per_day[2]
            self.regret_c0_ts.append(regret_c0)
            self.regret_c1_ts.append(regret_c1)
            self.regret_c2_ts.append(regret_c2)

            self.price_ev_per_day_ts.append(price)
