import numpy as np
from scipy.special import erfc
from learners.learner import Learner

#Gaussian Thompson Sampler learner
class TS_Learner(Learner):
    def __init__(self, n_arms, safety_t = 0.2):
        super().__init__(n_arms)
        #Safety parameter to avoid negative revenue
        self.safety_t = safety_t
        #Mean and Std
        self.normal_parameters = np.zeros((self.n_arms, 2))

    def pull_arm(self):
        if self.t < (2*self.n_arms): return (self.t % self.n_arms)

        #Safety constraint
        #Alternative implementation safety constraints using MATH         
        arg_max = []
        for p in self.normal_parameters : 
            if 0.5 * erfc(p[0]/(np.sqrt(2)*p[1])) > self.safety_t :
                arg_max.append(0.0) 
            else : 
                arg_max.append(np.random.normal(p[0], p[1]))
    
        return np.argmax(arg_max)

    def update(self, pulled_arm, reward):
        self.t += 1 
        times_pulled = max(1, len(self.rewards_per_arm[pulled_arm]))
        self.update_observations(pulled_arm, reward)
        #Empirical mean
        self.normal_parameters[pulled_arm, 0] = (self.normal_parameters[pulled_arm, 0] * (times_pulled - 1) + reward ) / times_pulled
        #Empirical std
        self.normal_parameters[pulled_arm, 1] = np.sqrt(sum((self.normal_parameters[pulled_arm, 0] - self.rewards_per_arm[pulled_arm])**2)/times_pulled**2)