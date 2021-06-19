import numpy as np
from tqdm import tqdm
from environment import Environment
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import matplotlib.pyplot as plt 

opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
opt_price = 6.321089806558111

class Experiment5:

    def __init__(self):
        #Variable initializationion
        self.price =  opt_price
        self.T = 365
        n_bids = 10 
        self.bids_space = np.linspace(1.0, 10.0, n_bids)
        self.ts_rewards_per_experiment = np.ndarray([])
        self.sols = []
        self.regret = []

        #Environment and Learner Creation
        self.env = Environment(noise_variance= 0.05)
        self.ts_learner = TS_Learner(n_arms= n_bids)

    def run(self):
        for t in tqdm(range(self.T)):
            #TS learner 
            bid_indx = self.ts_learner.pull_arm()
            bid = self.bids_space[bid_indx]
            bids = [bid,bid,bid]
            self.sols.append(bid)
            reward = self.env.round(bids,self.price)
            self.ts_learner.update(bid_indx, sum(reward))
            self.regret.append(sum(self.env.round(opt_bids, opt_price, noise = False)) - sum(reward))

    def showRegret(self):  
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(self.regret , 'r')
        plt.legend(["TS" ])
        plt.show()