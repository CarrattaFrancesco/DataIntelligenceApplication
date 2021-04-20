import numpy as np
from customerManager import CustomerManager

class Environment():

    def __init__(self,file_path="./config/customer_classes.json", noise_variance = 0.05):
        self.cManager = CustomerManager(file_path = file_path, noise_variance= noise_variance  )

    def round(self,bids,price, noise = True):
        #return an array (Uc1, Uc2, Uc3)
        return self.cManager.revenue(bids, price, noise= noise)

# enviroment that take in consideration just one class
class EnvironmentSingleClass(Environment):
    def round(self,bids,price,c_id, noise = True):
        #return an array (Uc1, Uc2, Uc3)
        return self.cManager.revenueSingleClass(bids, price,c_id, noise= noise)