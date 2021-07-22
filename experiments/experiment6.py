import numpy as np
from tqdm import tqdm
from environment import Environment
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import matplotlib.pyplot as plt 

opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
opt_price = 6.321089806558111

class Experiment6:

    def __init__(self):
        #Variable initialization
        self.T = 365
        self.regret = []
        self.sols = []
        self.n_bids = 10 
        n_prices = 13 
        n_arms = self.n_bids*n_prices
        self.bids_space = np.linspace(1.0, 10.0,self.n_bids)
        self.price_space = np.linspace(3.0, 15.0, n_prices)

        self.bids = []
        self.prices = []

        #Object Initialization
        self.env = Environment(noise_variance= 0.05)
        self.ts_learner = TS_Learner(n_arms = n_arms)
        self.ts_rewards_per_experiment = np.ndarray([])

    def run(self):
        for t in tqdm(range(self.T)):
            arm_indx = self.ts_learner.pull_arm()
            #Arm to bid, price conversion 
            bid = self.bids_space[arm_indx % self.n_bids]
            price = self.price_space[int(np.floor(arm_indx / self.n_bids))]
            #Storing Pulled arm
            bids = [bid,bid,bid]
            self.sols.append([bid,price])
            reward = self.env.round(bids,price)

            self.bids.append(bids)
            self.prices.append(price)

            #Updating
            self.ts_learner.update(arm_indx, sum(reward))

            self.regret.append(sum(self.env.round(opt_bids, opt_price, noise = False)) - sum(reward))

    def showRegret(self):
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(self.regret , 'r')
        plt.legend(["TS" ])
        plt.show()

    def show_bids_per_class(self):
        bids = np.array(self.bids) 


        fig, axs = plt.subplots(3)
        fig.suptitle('Estimated bids with TS')
        for i in range(3):
            axs[i].plot(bids[:,i],'y')
            axs[i].hlines(opt_bids[i], 0, 365, 'g', linestyles='dashed')
            axs[i].set(ylabel="Class "+ str(i))
        fig.legend(["Bids","Optimal Bids"])
    
    def show_prices(self):
        prices = np.array(self.prices) 

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("prices")
        plt.plot(prices , 'y')
        plt.hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        plt.legend(["Price","Optimal Price" ])
        plt.show()
