import json
from customer import  Customer
import numpy as np

class CustomerManager():

    def __init__(self, file_path, noise_variance = 0.05) :
        self.customers = [] 
        self.noise_variance = noise_variance
        json_file = open(file_path)
        self.classes = json.load(json_file)["classes"]
        for c in self.classes: 
            self.customers.append(Customer(id = c["class_id"],
                                          clicks_coefficient = c["clicks_coefficient"] ,
                                          slope_coefficient = c["slope_coefficient"],
                                          conversion_rate_matrix = c["conversion_rate_matrix"], 
                                          comeback_vector = c["comeback_vector"]))
        
    def __get_by_id(self, id):
        for c in self.customers:
            if c.class_id == id:
                return c
        raise Exception("Invalid ID")
    

    def clicks(self,class_id, bid, noise = True):
        res = self.__get_by_id(class_id).clicks(bid)
        if not noise : 
            return res 
        return  np.floor( np.clip(res + np.random.normal(0, abs(res * self.noise_variance)  ), a_min = 0, a_max = None) )

    def cost_per_click(self, class_id, bid, noise = True): 
        res = self.__get_by_id(class_id).cost_per_click(bid) 
        if not noise : 
            return res 
        return np.clip(res + np.random.normal(0, abs(res * self.noise_variance)  ), a_min = 0, a_max = None)

    def comeback_probability(self, class_id, times, noise = True):
        res = self.__get_by_id(class_id).comeback_probability(times)
        if not noise : 
            return res 
        return np.clip( res + np.random.normal(0, abs(res * self.noise_variance)  ), a_min = 0, a_max = 1)

    def conversion_rate(self, class_id, price, noise = True):
        res = self.__get_by_id(class_id).conversion_rate(price)
        if not noise : 
            return res 
        return np.clip( res + np.random.normal(0, abs(res * self.noise_variance)  ), a_min = 0, a_max = 1)


    def sold_items(self, class_id, bid, price, noise = True):
        res = self.__get_by_id(class_id).sold_items(bid = bid, price = price)
        if not noise :
            return res 
        return np.clip( res + np.random.normal(0, abs(res * self.noise_variance) ), a_min = 0, a_max = None)
    
    def revenue(self, bids, price, noise=True):
        revenue = []
        for c in self.classes:
            c_id = int(c["class_id"])
            revenue.append((self.sold_items(c_id, bids[c_id], price,noise=noise) * price) - (self.clicks(c_id, bids[c_id],noise=noise) * self.cost_per_click(c_id, bids[c_id],noise=noise)))
        return revenue

    def revenueSingleClass(self,bids,price,c_id,noise=True):
        return (self.sold_items(c_id, bids[c_id], price,noise=noise) * price) - (self.clicks(c_id, bids[c_id],noise=noise) * self.cost_per_click(c_id, bids[c_id],noise=noise))