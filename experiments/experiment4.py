import numpy as np
from environment import Environment
from learners.ucb_learner import UCB1
from learners.ts_learner import TS_Learner
import matplotlib.pyplot as plt 

class Experiment4:

    def __init__(self):
        self.env = Environment()
        n_arms = 13

        #Part1
        self.ucb1_learner_c0 = UCB1(n_arms = n_arms)
        self.ucb1_learner_c1 = UCB1(n_arms = n_arms)
        self.ucb1_learner_c2 = UCB1(n_arms = n_arms)
        self.ts_learner_c0 = TS_Learner(n_arms = n_arms)
        self.ts_learner_c1 = TS_Learner(n_arms = n_arms)
        self.ts_learner_c2 = TS_Learner(n_arms = n_arms)

        self.regret_c0_ucb = []
        self.regret_c1_ucb = []
        self.regret_c2_ucb = []

        self.regret_c0_ts = []
        self.regret_c1_ts = []
        self.regret_c2_ts = []

        
        self.price_ev_per_day_ucb = []
        self.price_ev_per_day_ts = []

        #Part2 
        self.ucb1_learner_c0_part2 = UCB1(n_arms = n_arms)
        self.ucb1_learner_c12_part2 = UCB1(n_arms = n_arms)
        self.ts_learner_c0_part2 = TS_Learner(n_arms = n_arms)
        self.ts_learner_c12_part2 = TS_Learner(n_arms = n_arms)
        self.regret_c0_ucb_part2 = []
        self.regret_c12_ucb_part2 = []
        self.regret_c0_ts_part2 = []
        self.regret_c12_ts_part2 = []

        self.price_ev_per_day_ucb_part2 = []
        self.price_ev_per_day_ts_part2 = []

        self.T = 365
        self.prices = np.linspace(3.0, 15.0, n_arms)

        
    
    def run(self):
        opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
        opt_price = 6.321089806558111
        bids = opt_bids 
        for _ in range(self.T):
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
            
            reward_optimal = self.env.round(opt_bids, opt_price, noise= False)
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

    def run_part2(self):
        opt_bids = [3.8622484787564275 , 2.1216094606111944,  2.347134281066495]
        opt_price = 6.321089806558111
        bids = opt_bids 
        for _ in range(self.T):
            #UCB1 learner
            price_c0_idx = self.ucb1_learner_c0_part2.pull_arm()
            price_c0 = self.prices[price_c0_idx]
            price_c12_idx = self.ucb1_learner_c12_part2.pull_arm()
            price_c12 = self.prices[price_c12_idx]
            
            reward_per_day = []
            price = [price_c0, price_c12]
            for i in range(len(price)):
                if i == 0 : 
                    p = price[0]
                    reward = self.env.round(bids, p)
                    reward_per_day.append(reward[0])
                else :
                    p = price[i]
                    reward = self.env.round(bids, p)
                    reward_per_day.append(reward[1]+reward[2])
                    

            self.ucb1_learner_c0_part2.update(price_c0_idx, reward_per_day[0])
            self.ucb1_learner_c12_part2.update(price_c12_idx, reward_per_day[1])
            
            reward_optimal = self.env.round(opt_bids, opt_price, noise= False)
            regret_c0 = reward_optimal[0] - reward_per_day[0]
            regret_c12 = (reward_optimal[1]+reward_optimal[2]) - reward_per_day[1]
            self.regret_c0_ucb_part2.append(regret_c0)
            self.regret_c12_ucb_part2.append(regret_c12)
            
            self.price_ev_per_day_ucb_part2.append(price)

            #TS learner
            price_c0_idx = self.ts_learner_c0_part2.pull_arm()
            price_c0 = self.prices[price_c0_idx]
            price_c12_idx = self.ts_learner_c12_part2.pull_arm()
            price_c12 = self.prices[price_c12_idx]
            
            reward_per_day = []
            price = [price_c0, price_c12]
            for i in range(len(price)):
                if i == 0 : 
                    p = price[i]
                    reward = self.env.round(bids, p)
                    reward_per_day.append(reward[0])
                else :    
                    p = price[1]
                    reward = self.env.round(bids, p)
                    reward_per_day.append(reward[1]+reward[2])
                

            self.ts_learner_c0_part2.update(price_c0_idx, reward_per_day[0])
            self.ts_learner_c12_part2.update(price_c12_idx, reward_per_day[1])
            
            reward_optimal = self.env.round(opt_bids, opt_price, noise= False)
            regret_c0 = reward_optimal[0] - reward_per_day[0]
            regret_c12 = (reward_optimal[1]+reward_optimal[2]) - reward_per_day[1]
            self.regret_c0_ts_part2.append(regret_c0)
            self.regret_c12_ts_part2.append(regret_c12)
            
            self.price_ev_per_day_ts_part2.append(price)

    def show_UCB_regret_separated_classes(self):
        plt.figure(0)
        plt.xlabel("Day")
        plt.ylabel("Regret")
        plt.plot(self.regret_c0_ucb,'r', label="UCB-c0")
        plt.plot(self.regret_c1_ucb, 'b', label="UCB-c1")
        plt.plot(self.regret_c2_ucb, 'g', label="UCB-c2")
        plt.title("Step 4 - UCB1 regret")
        plt.legend()
        plt.show()
    
    def show_TS_regret_separated_classes(self):
        plt.figure(1)
        plt.xlabel("Day")
        plt.ylabel("Regret")
        plt.plot(self.regret_c0_ts,'r', label="TS-c0")
        plt.plot(self.regret_c1_ts, 'b', label="TS-c1")
        plt.plot(self.regret_c2_ts, 'g', label="TS-c2")
        plt.title("Step 4 - TS regret")
        plt.legend()
        plt.show()

    def show_UCBvsTS_separeted_classes(self):
        fig, axs = plt.subplots(3)
        fig.suptitle('Regret: UCB vs TS per class')
        axs[0].plot(self.regret_c0_ucb,'y', label="UCB-c0", linestyle="dashed")
        axs[0].plot(self.regret_c0_ts,'r', label="TS-c0")
        axs[0].legend()
        axs[1].plot(self.regret_c1_ucb,'y', label="UCB-c1", linestyle="dashed")
        axs[1].plot(self.regret_c1_ts,'b', label="TS-c1")
        axs[1].legend()
        axs[2].plot(self.regret_c2_ucb,'y', label="UCB-c2", linestyle="dashed")
        axs[2].plot(self.regret_c2_ts,'g', label="TS-c2")
        axs[2].legend()

    def show_price_per_class(self):
        YY_ucb = np.array(self.price_ev_per_day_ucb) 
        YY_ts = np.array(self.price_ev_per_day_ts)
        opt_price = 6.321089806558111

        fig, axs = plt.subplots(3)
        fig.suptitle('Estimated price: UCB vs TS per class')
        axs[0].plot(YY_ucb[:,0],'y', linestyle="dashed")
        axs[0].plot(YY_ts[:,0],'r')
        axs[0].hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        axs[0].set(ylabel="Class 0")
        axs[1].plot(YY_ucb[:,1],'y', linestyle="dashed")
        axs[1].plot(YY_ts[:,1],'r')
        axs[1].hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        axs[1].set(ylabel="Class 1")
        axs[2].plot(YY_ucb[:,2],'y', linestyle="dashed")
        axs[2].plot(YY_ts[:,2],'r')
        axs[2].hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        axs[2].set(ylabel="Class 2")
        fig.legend(["UCB","TS","Optimal Price"])

    def show_regret_UCB_part2(self):          
        plt.figure(0)
        plt.xlabel("Day")
        plt.ylabel("Regret")
        plt.plot(self.regret_c0_ucb_part2,'r', label="UCB-c0")
        plt.plot(self.regret_c12_ucb_part2, 'b', label="UCB-c1+c2")
        plt.title("Step 4 - UCB1 regret")
        plt.legend()
        plt.show()

    def show_regret_TS_part2(self):
        plt.figure(1)
        plt.xlabel("Day")
        plt.ylabel("Regret")
        plt.plot(self.regret_c0_ts_part2,'r', label="TS-c0")
        plt.plot(self.regret_c12_ts_part2, 'b', label="TS-c1+c2")
        plt.title("Step 4 - TS regret")
        plt.legend()
        plt.show()

    def show_UCBvsTS_part2(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Regret: UCB vs TS per class (c1+c2 aggregated)')
        axs[0].plot(self.regret_c0_ucb_part2,'y', label="UCB-c0", linestyle="dashed")
        axs[0].plot(self.regret_c0_ts_part2,'r', label="TS-c0")
        axs[0].legend()
        axs[1].plot(self.regret_c12_ucb_part2,'y', label="UCB-c1+c2", linestyle="dashed")
        axs[1].plot(self.regret_c12_ts_part2,'b', label="TS-c1+c2")
        axs[1].legend()

    def estimated_price_part2(self):
        opt_price = 6.321089806558111
        YY_ucb = np.array(self.price_ev_per_day_ucb_part2) 
        YY_ts = np.array(self.price_ev_per_day_ts_part2)
        fig, axs = plt.subplots(2)
        fig.suptitle('Estimated price: UCB vs TS per class (c1+c2 aggregated)')
        axs[0].plot(YY_ucb[:,0],'y', linestyle="dashed")
        axs[0].plot(YY_ts[:,0],'r')
        axs[0].hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        axs[0].set(ylabel="Class 0")
        axs[1].plot(YY_ucb[:,1],'y', linestyle="dashed")
        axs[1].plot(YY_ts[:,1],'r')
        axs[1].hlines(opt_price, 0, 365, 'g', linestyles='dashed')
        axs[1].set(ylabel="Class 1 + Class 2")
        fig.legend(["UCB","TS","Optimal Price"])