import numpy as np

class Customer():

    def __init__(self,id, clicks_coefficient, slope_coefficient, conversion_rate_matrix, comeback_vector):
        self.class_id = id
        self.clicks_coefficient = clicks_coefficient
        self.slope_coefficient = slope_coefficient
        self.conversion_rate_matrix = conversion_rate_matrix
        self.comeback_vector = comeback_vector
        self.expected_returns = self.__expected_returns()
    
    def clicks(self, bid):
        #Calculates the clicks as function of the bid for this customer type
        return np.floor( self.clicks_coefficient * (1 - np.exp(-self.slope_coefficient*bid)) )

    def cost_per_click(self, bid): 
        #Computes the cost per click as 
        return bid #TODO: It's not actually the bid, but something less

    def comeback_probability(self, times):
        #Return a probability of coming back for the user based on the class, extreme points are default.
        if times < 1:
            raise Exception("Value Error: Times can't be zero or negative.")
        if times >= 20: return 0.00
        return self.comeback_vector[times - 1]

    def conversion_rate(self, price):
        #Calculate the conversion rate given the conversion rate matrix linearly interpolating the points.
        #First element of conversion matrix should be [0, 1.0]
        if price <= 0 :
            raise Exception("Value Error: Price can't be zero or negative.")
        if price < 3 or price > 15 : return 0.0
        
        i = 0
        while self.conversion_rate_matrix[i][0] < price and i < len(self.conversion_rate_matrix)-1:
            i+=1
        
        #Value calculation
        xa = self.conversion_rate_matrix[i-1][0]
        xb = self.conversion_rate_matrix[i][0]
        ya = self.conversion_rate_matrix[i-1][1]
        yb = self.conversion_rate_matrix[i][1]

        #Linear Iterpolation
        return (((price - xb)/(xa -xb)) * ya) - (((price - xa)/(xa -xb)) * yb)

    def sold_items(self, bid, price):
        return np.clip(self.clicks(bid) * self.conversion_rate(price) * self.expected_returns, 0, None)

    
    def __expected_returns(self):
        e = 0
        for i in range(1, 20):
            ##Computes the times that this customer type will come back on average
            e += self.comeback_probability(i) * 1  
        return e
