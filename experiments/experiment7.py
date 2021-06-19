import numpy as np
from tqdm import tqdm
from environment import EnvironmentSingleClass
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import matplotlib.pyplot as plt 


opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
opt_price = 6.321089806558111

class Experiment7:

    def __init__(self):
        #Variable initialization
        self.T = 365
        self.n_classes = 3
        self.n_bids = 10 
        n_prices = 13 
        n_arms = self.n_bids*n_prices
        self.bids_space = np.linspace(1.0, 10.0, self.n_bids)
        self.price_space = np.linspace(3.0, 15.0, n_prices)
        self.bids = np.ones(3)
        self.prices = np.ones(3)
        self.regrets =[[],[],[]]
        self.sols = [[],[],[]] #Tryed Solutions

        #Object Initialization 
        self.env = EnvironmentSingleClass(noise_variance= 0.05)
        self.ts_learners= [TS_Learner(n_arms = n_arms),TS_Learner(n_arms = n_arms),TS_Learner(n_arms = n_arms)]
        
    def run(self):
        
        for t in tqdm(range(self.T)):
            #TS learner 
            for c in range(self.n_classes):
                arm_indx = self.ts_learners[c].pull_arm()
                #Arm to bid, price conversion 
                self.bids[c] = self.bids_space[arm_indx % self.n_bids]
                self.prices[c] = self.price_space[int(np.floor(arm_indx / self.n_bids))]
                #Storing found solution
                self.sols[c].append([self.bids[c],self.prices[c]])
                reward = self.env.round(self.bids[c],self.prices[c],c_id = c)
                self.ts_learners[c].update(arm_indx, reward)
                self.regrets[c].append(self.env.round(opt_bids[c], opt_price,c, noise = False) - reward)

    def showRegret(self):
        colors = ['r','g','b']

        for c in range(3):
            plt.figure(0)
            plt.xlabel("t")
            plt.ylabel("Regret")
            plt.plot(self.regrets[c] , colors[c])
            plt.legend(["Class_"+str(c) ])
            plt.show()