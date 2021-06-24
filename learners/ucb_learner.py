import numpy as np
from learners.learner import Learner

# Upper-Confidence Bound Learner
class UCB1(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            return  self.t 
        upper_bound = self.empirical_means + self.confidence
        return np.argmax(upper_bound)
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards.append(reward)
        times_pulled = len(self.rewards_per_arm[pulled_arm])
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (times_pulled-1) + reward)/times_pulled
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]) ) 
            self.confidence[a] = (2*np.log(self.t) /number_pulled)**0.5
       