import json
from customer import  Customer
import numpy as np

class CustomerManager():

    def __init__(self, file_path, noise_variances = np.ones(shape = (4))) :
        self.customers = [] 
        self.noise_variances = noise_variances
        json_file = open(file_path)
        classes = json.load(json_file)["classes"]
        for c in classes: 
            self.customers.append(Customer(id = c["class-id"],
                                          clicks_coefficient = c["properties"]["clicks_coefficient"] , 
                                          cost_coefficient = c["properties"]["cost_coefficient"],
                                          conversion_rate_matrix = c["properties"]["conversion_rate_matrix"], 
                                          comeback_vector = c["properties"]["comeback_vector"]))
    
    def __get_by_id(self, id):
        for c in self.customers:
            if c.class_id == id:
                return c
        raise Exception("Invalid ID")
    

    def clicks(self,class_id, bid):
        return np.clip(self.__get_by_id(class_id).clicks(bid) + np.random.normal(0, self.noise_variances[0]), a_min = 0, a_max = None)

    def cost_per_click(self, class_id, bid): 
        return np.clip(self.__get_by_id(class_id).cost_per_click(bid) + np.random.normal(0, self.noise_variances[1]), a_min = 0, a_max = None)

    def comeback_probability(self, class_id, times):
        return np.clip(self.__get_by_id(class_id).comeback_probability(times) + np.random.normal(0, self.noise_variances[2]), a_min = 0, a_max = 1)

    def conversion_rate(self, class_id, price):
        return np.clip(self.__get_by_id(class_id).conversion_rate(price) + np.random.normal(0, self.noise_variances[3]), a_min = 0, a_max = 1)
