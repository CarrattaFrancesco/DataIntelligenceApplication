import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy.optimize as opt


class UCB1_Learner():

    def __init__(self):
        self.bid = np.linspace(1.0, 10.0, 10)
        self.price = np.linspace(3.0, 15.0, 12)
        self.t = 0 
        
        self.n_arms = self.bid.size * self.price.size
        self.collected_rewards = []

        #Arms are an ndarray, in the implementation of the prof he stores all the reward, maybe  it's useless
        #We store the number of pulls 
        self.rewards_per_arms_c1 = np.zeros((self.bid.size, self.price.size))
        self.rewards_per_arms_c2 = np.zeros((self.bid.size, self.price.size))
        self.rewards_per_arms_c3 = np.zeros((self.bid.size, self.price.size))       

        #Empirical means for each class
        self.empirical_means_c1 = np.zeros((self.bid.size, self.price.size))
        self.empirical_means_c2 = np.zeros((self.bid.size, self.price.size))
        self.empirical_means_c3 = np.zeros((self.bid.size, self.price.size))
        
        #Confidence for each class
        self.confidence_c1 = np.zeros((self.bid.size, self.price.size))
        self.confidence_c2 = np.zeros((self.bid.size, self.price.size))
        self.confidence_c3 = np.zeros((self.bid.size, self.price.size))
        
    def __get_arm_indexes(self, bid, price):
        return (int(bid - 1), int(price - 3))

    def update(self, bids, price, rewards): 
        
        self.t += 1
        #Pulled arm of each class is a tuple of (bid, price)
        pulled_arm_c2 = self.__get_arm_indexes(bids[1], price)
        pulled_arm_c3 = self.__get_arm_indexes(bids[2], price)
        pulled_arm_c1 = self.__get_arm_indexes(bids[0], price)

        #Rewards has to be a tuple of 3 elements, the reward of each class   
        self.rewards_per_arms_c1[pulled_arm_c1] += 1
        self.rewards_per_arms_c2[pulled_arm_c2] += 1
        self.rewards_per_arms_c3[pulled_arm_c3] += 1

        #Stores the rewards obtained
        self.collected_rewards.append(rewards)

        #Update of the emipirical means
        self.empirical_means_c1[pulled_arm_c1] = (self.empirical_means_c1[pulled_arm_c1] * (self.t-1) + rewards[0]) /self.t
        self.empirical_means_c2[pulled_arm_c2] = (self.empirical_means_c2[pulled_arm_c2] * (self.t-1) + rewards[1]) /self.t
        self.empirical_means_c3[pulled_arm_c3] = (self.empirical_means_c3[pulled_arm_c3] * (self.t-1) + rewards[2]) /self.t

        #checks and build the confidence bounds 
        for b in range(self.bid.size):
            for p in range(self.price.size):
                number_pulled_c1 = max(1 , self.rewards_per_arms_c1[b, p])
                number_pulled_c2 = max(1 , self.rewards_per_arms_c2[b, p])
                number_pulled_c3 = max(1 , self.rewards_per_arms_c3[b, p])
                self.confidence_c1[b,p] = (2*np.log(self.t)/ number_pulled_c1) ** 0.5
                self.confidence_c2[b,p] = (2*np.log(self.t)/ number_pulled_c2) ** 0.5
                self.confidence_c3[b,p] = (2*np.log(self.t)/ number_pulled_c3) ** 0.5


    def pull_arm(self):
        if self.t < self.n_arms:
            bid = np.floor(self.t / self.price.size)
            return ([bid + 1,bid +1 ,bid +1] , self.t % self.price.size + 3)

        
        upper_bounds_c1 = self.empirical_means_c1 + self.confidence_c1
        upper_bounds_c2 = self.empirical_means_c2 + self.confidence_c2
        upper_bounds_c3 = self.empirical_means_c3 + self.confidence_c3

        tot_rewards = np.zeros(self.price.size)

        bid_1 = np.zeros(self.price.size)  
        bid_2 = np.zeros(self.price.size)
        bid_3 = np.zeros(self.price.size)

        for p in range(self.price.size):

            ub1 = np.array(upper_bounds_c1[:,p], dtype=np.int8)
            ub2 = np.array(upper_bounds_c2[:,p], dtype=np.int8)
            ub3 = np.array(upper_bounds_c3[:,p], dtype=np.int8)
            
            #This generates the index with the best bid for price p 
            bid_1[p] = np.random.choice(np.where(ub1 == ub1.max())[0])
            bid_2[p] = np.random.choice(np.where(ub2 == ub2.max())[0])
            bid_3[p] = np.random.choice(np.where(ub3 == ub3.max())[0])

            tot_rewards[p] = upper_bounds_c1[int(bid_1[p]), p] + upper_bounds_c2[int(bid_2[p]), p] + upper_bounds_c3[int(bid_3[p]), p]
        
        best = np.random.choice(np.where(tot_rewards == tot_rewards.max())[0])
        
        #Returns the best combination of bids and price (value)
        p = best + 3 
        bids = [int(bid_1[best]) + 1, int(bid_2[best]) + 1, int(bid_3[best]) + 1]

        return (bids, p)


class GP_Learner():

    def __init__(self,   theta = 1.0, l = 1.0, noise_std = 5.0):
        self.theta = theta 
        self.l = l
        self.noise_std = noise_std 
        self.classes = ['0', '1', '2']
        self.kernel = C(theta, (1e-2, 1e2)) * RBF(1, (1e-2, 1e2)) 
        self.revenue = GaussianProcessRegressor(kernel = self.kernel, alpha = noise_std**2, normalize_y = False, n_restarts_optimizer = 3)
        
        self.revenues_observed  = np.zeros(1)
        self.x_obs = np.zeros(4)
        self.opt_res = np.array([1.0, 1.0, 1.0, 4.5])
        self.error = []
        

    def fit(self, bids, price, revenue):
        
        self.revenues_observed = np.append(self.revenues_observed, revenue)

        new_x_obs = np.append([], [bids[0],bids[1],bids[2],price])
        self.x_obs = np.vstack((self.x_obs, new_x_obs))

        X = self.x_obs
        Y = self.revenues_observed.ravel()
        
        if(X.shape[0] >= 2):
            self.revenue.fit(X, Y)


    def optimize(self): 
        bids = np.linspace(1,10,10)
        prices = np.linspace(3,15,13)
        x_pred = self.x_obs
        self.y_pred = self.revenue.predict(x_pred)
        
        mse = sum((self.y_pred - self.revenues_observed) ** 2) / self.y_pred.size
        
        self.error.append(mse)

        #Exploration part
        rnd_res = np.array([np.random.choice(bids), np.random.choice(bids), np.random.choice(bids), np.random.choice(prices)])
        
        if np.random.uniform() < 1000/((self.y_pred.size)**2) or self.y_pred.size < 100 :
            return rnd_res[:3] , rnd_res[3]
       
        #Exploitation
        x0 = np.array([np.random.choice(bids), np.random.choice(bids), np.random.choice(bids), np.random.choice(prices)])

        self.opt_res = opt.minimize(self.objective, x0, method = 'Powell').x  

        for i in range(4):
            self.opt_res[i] = np.floor(self.opt_res[i])

        return (self.opt_res[:3], self.opt_res[3])  
        


    def objective(self, x):
        bids = x[:3]
        price = x[3]

        bids_values = np.linspace(1,10,10)
        price_values = np.linspace(3,15,13)

        #Checks integrity
        if price <= 0 or price > 15 : return 0.0 
        if any(b < 1 or b > 10 for b in bids): return 0.0

        #Data to predict
        x1 = np.array([np.random.choice(bids_values), np.random.choice(bids_values), np.random.choice(bids_values), np.random.choice(price_values)])
    
        xs = np.vstack((x, x1))

        res = self.revenue.predict(xs)

        return -res[0]

    def objectiveMin(self, x):
        bids = x[:3]
        price = x[3]

        bids_values = np.linspace(1,10,10)
        price_values = np.linspace(3,15,13)

        #Checks integrity
        if price <= 0 or price > 15 : return 0.0 
        if any(b < 1 or b > 10 for b in bids): return 0.0

        #Data to predict
        x1 = np.array([np.random.choice(bids_values), np.random.choice(bids_values), np.random.choice(bids_values), np.random.choice(price_values)])
    
        xs = np.vstack((x, x1))

        res = self.revenue.predict(xs)

        return res[0]
        


         


        