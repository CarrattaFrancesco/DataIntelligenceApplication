import numpy as np
from customerManager import CustomerManager

class Env_1():

    def __init__(self,file_path="./config/customer_classes.json"):
        self.cManager = CustomerManager(file_path = file_path)
        self.bids = np.linspace(1.0, 10.0, 20)
        self.price = np.linspace(3.0, 15.0, 20)

    def generateInput(self):
        res = {
            'price': np.random.choice(self.price, 1)[0],
            'bid_class_0':  np.random.choice(self.bids, 1)[0],
            'bid_class_1':  np.random.choice(self.bids, 1)[0],
            'bid_class_2':  np.random.choice(self.bids, 1)[0]
        }

        return res

    def round(self,bid,price,c_id):
        clicks = self.cManager.clicks(class_id = c_id, bid = bid)
        sold_items = self.cManager.sold_items(class_id = c_id, bid = bid, price = price)
        revenue = (price * sold_items) - (self.cManager.cost_per_click(class_id=c_id, bid = bid) * clicks)
        
        dic = {
            'id': c_id,
            'clicks' : clicks,
            'revenue': revenue,
            'sold_items': sold_items
            
        }
        
        return dic

    '''
    Only for testing purpose
    '''
    def roundClicks(self, bid, c_id):
        return self.cManager.clicks(class_id = c_id, bid = bid)
   
