import numpy as np
from matplotlib import pyplot as plt 
from learners.gp_learner import GP_Learner
from environment import Environment
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt 
warnings.filterwarnings("ignore")



class Experiment2:

    def __init__(self,delay=0):
        self.opt_bids = [3.9416202240277745 , 3.0,  4.0]
        self.opt_price = 7

        self.bids = np.array([1.0, 1.0, 1.0])
        self.price = 4.5      #Initial Price
        self.T = 365 

        self.env = Environment()
        self.gp_learner = GP_Learner()
        self.regrets = [] 
        self.sols = []
        self.delay= delay
        self.input = {
            'x':[],
            'outcome':[]
        }

    
    def run(self):
        for t in tqdm(range(self.T)):
            outcome = sum(self.env.round(self.bids, self.price))
            self.input['outcome'].append(outcome)

            x = np.append(self.bids,self.price)
            self.input['x'].append(x)
            
            if t >= self.delay:
                self.gp_learner.fit(self.input['x'][t-self.delay], self.input['outcome'][t-self.delay])
                self.bids, self.price = self.gp_learner.optimize()
            else:
                self.bids = np.array([1.0, 1.0, 1.0])
                self.price = 4.5 
            
            
            self.sols.append((self.bids, self.price))
            self.regrets.append(sum(self.env.round(self.opt_bids, self.opt_price, noise = False)) - outcome)

    def showRegret(self):
        
        X = np.arange(0,365)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(X, self.regrets , 'r')

        plt.show()


    def showMSE(self):
        X = np.arange(0,365- self.delay)
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("MSE_revenue")
        plt.plot(X, self.gp_learner.errors , 'r')

        plt.show()

    def foundSolution(self):
        return self.sols[-1]