import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import scipy.optimize as opt

class GP_Learner():

    def __init__(self,   theta = 1.0, l = 1.0, noise_std = 5.0):
        self.theta = theta 
        self.l = l
        self.noise_std = noise_std 
        self.classes = ['0', '1', '2']
        self.kernel = C(theta, (1e-2, 1e2)) * RBF(1, (1e-2, 1e2)) 
        self.revenue = GaussianProcessRegressor(kernel = self.kernel, alpha = noise_std**2, normalize_y = False, n_restarts_optimizer = 3)
        
        self.y_obs  = np.zeros(1)
        self.x_obs = np.zeros(4)
        self.opt_res = np.zeros(4)
        self.errors = []
        

    def fit(self, x, y):
        
        self.y_obs = np.append(self.y_obs, y)
        self.x_obs = np.vstack((self.x_obs, x))

        X = self.x_obs
        Y = self.y_obs.ravel()
        
        if(X.shape[0] >= 2):
            self.revenue.fit(X, Y)


    def optimize(self): 
        bids = np.linspace(1,10,10)
        prices = np.linspace(3,15,13)

        self.y_pred, sigma = self.revenue.predict(self.x_obs, return_std=True)
        mse = sum((self.y_pred - self.y_obs) ** 2) / self.y_pred.size
        self.errors.append(mse)

        #Exploration part
        if np.random.uniform() < 1000/((self.y_pred.size)**2) and self.y_pred.size < 100:
            rnd_res = np.array([np.random.choice(bids), np.random.choice(bids), np.random.choice(bids), np.random.choice(prices)])
            return rnd_res[:3], rnd_res[3]

        #Exploitation
        x0 = np.array([np.random.choice(bids), np.random.choice(bids), np.random.choice(bids), np.random.choice(prices)])

        x = opt.minimize(self.objective, x0, method = 'Powell').x  

        if x[3] >= 3 and x[3] <= 15 and all(i <= 10 and i >= 1 for i in x[:3]):  
            self.opt_res = x
        
        for i in range(4):
            self.opt_res[i] = round(self.opt_res[i])

        #Avoid convergence in suboptimal points, explore close price set
        if  np.random.uniform() < 0.02 and self.y_pred.size > 100:
            
            return (
               [np.clip(self.opt_res[0] + np.random.choice([-1,0, 1]), 1 ,10),
                np.clip(self.opt_res[1] + np.random.choice([-1,0, 1]), 1 ,10),
                np.clip(self.opt_res[2] + np.random.choice([-1,0, 1]), 1 ,10)],
                np.clip(self.opt_res[3] + np.random.choice([-1,0, 1]), 3 ,15)
            )

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

   
         


        