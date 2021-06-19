import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from customerManager import CustomerManager
customer_path = "./config/customer_classes.json"
cManager = CustomerManager(file_path = customer_path)

def objective( x):
    bids = x[:3]
    price = x[3]

    #Checks integrity
    if price < 3 or price > 15 : return 0.0 
    if any(b < 1 or b > 10 for b in bids): return 0.0


    res = cManager.revenue(bids,price,noise=False)

    return -sum(res)

def experiment1run():
    # optimize
    b_p = (3.0,15.0)
    b_b = (1.0 , 10.0)
    bnds = (b_b, b_b, b_b, b_p)

    method = 'Powell'

    max_value = 0
    best_method = 'none'
    best_x = None



    bids = np.linspace(1,10,10) 
    prices = np.linspace(3,15,13)

    X_tmp = []
    X= []
    n_starting_points = 50
    #generate random starting point
    for b1 in bids:
        for b2 in bids:
            for b3 in bids:
                for p in prices:
                    X_tmp.append([b1,b2,b3,p])

    for i in range(n_starting_points):
        X.append(X_tmp[np.random.choice(len(X_tmp))])

    X= np.array(X)

    for x0 in tqdm(X):
        solution = opt.minimize(objective, x0, method = 'Powell') 
        x = solution.x

        if max_value < -objective(x):
            max_value = -objective(x)
            best_x = x

    print("\n")
    print("Best method is " + best_method + " with a value of " + str(max_value))
    print('Optimal Solution')
    print('bid 1 = ' + str(best_x[0]))
    print('bid 2 = ' + str(best_x[1]))
    print('bid 3 = ' + str(best_x[2]))
    print('price = ' + str(best_x[3]))