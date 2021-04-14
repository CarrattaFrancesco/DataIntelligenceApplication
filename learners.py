import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Learner_1():

    def __init__(self, noise = 1.0, theta = 1.0, l= 1.0):
        self.noise_std = noise
        self.theta = theta
        self.l = l
        

    def fit(self,X,Y):
        kernel = C(self.theta, (1e-05, 1e5)) * RBF(1, (1e-5, 1e5)) 
        gp = GaussianProcessRegressor(kernel = kernel, alpha = self.noise_std**2, normalize_y = False, n_restarts_optimizer = 10)
        gp.fit(X,Y)
        self.model = gp

    def predict(self,X, return_std = False):
        return self.model.predict(X, return_std = return_std)

