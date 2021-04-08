import numpy as np

class Customer():

    def __init__(self,id, clicks_coefficient, cost_coefficient, conversion_rate_matrix, comeback_vector):
        self.class_id = id
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
        #Return a probability of coming back for the user based on the class, extreme points are default.
        if times < 1:
            raise Exception("Value Error: Price can't be zero or negative.")
        if times >= 20: return 0.00
        return self.comeback_vector[times - 1]

    def conversion_rate(self, price):
        #Calculate the conversion rate given the conversion rate matrix linearly interpolating the points.
        #First element of conversion matrix should be [0, 1.0]
        if price <= 0 :
            raise Exception("Value Error: Price can't be zero or negative.")
        
        i = 0
        while self.conversion_rate_matrix[i][0] < price :
            i+=1
        
        #Value calculation
        xa = self.conversion_rate_matrix[i-1][0]
        xb = self.conversion_rate_matrix[i][0]
        ya = self.conversion_rate_matrix[i-1][1]
        yb = self.conversion_rate_matrix[i][1]

        #Linear Iterpolation
        return (((price - xb)/(xa -xb)) * ya) - (((price - xa)/(xa -xb)) * yb)

