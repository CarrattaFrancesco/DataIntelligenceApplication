import numpy as np
from customerManager import CustomerManager

class Environment():

    def __init__(self,file_path="./config/customer_classes.json"):
        self.cManager = CustomerManager(file_path = file_path, noise_variance= 0.01)

    def round(self,bids,price):
        #return an array (Uc1, Uc2, Uc3)
        return self.cManager.revenue(bids, price)

    