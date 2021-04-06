import numpy as np

class CustomerType():

    def __init__(self, clicks_coefficient, cost_coefficient, conversion_rate_matrix, comeback_vector):
        self.clicks_coefficient = clicks_coefficient
        self.cost_coefficient = cost_coefficient   
        self.conversion_rate_matrix = conversion_rate_matrix
        self.comeback_vector = comeback_vector
    
    def clicks(self, bid):
        #Calculates the clicks as function of the bid for this customer type
        return self.clicks_coefficient * (1 - np.exp(-bid)) 

    def cost_per_click(self, bid): 
        #Computes the cost per click
        return bid / (self.cost_coefficient * (1 - np.exp(-bid)))

    def comeback_probability(self, times):
        # Return a probability of coming back for the user based on the class, extreme points are default.
        if times <= 1: return 1.0
        if times > 14 and < 25 return 0.05
        if times >= 25 return 0.00
        return self.comeback_vector[times - 2]

    def conversion_rate(self, price):
        #TODO : similar thing of above, take the two value of the price above and below the given one, and iterpolate.
        return 0.5

