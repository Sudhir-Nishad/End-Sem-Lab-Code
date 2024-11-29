import numpy as np
from scipy.stats import poisson

MAX_BIKES = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT_FACTOR = 0.9


request_mean = [3, 4]  
return_mean = [3, 2]    

def calculate_reward_and_transition(state, action):
    x1, x2 = state
    net_move = action
    
    
    new_x1 = max(0, x1 - net_move) if net_move > 0 else min(MAX_BIKES, x1 + net_move)
    new_x2 = max(0, x2 + net_move) if net_move < 0 else min(MAX_BIKES, x2 - net_move)
    
    
    rentals1 = poisson.rvs(request_mean[0])
    rentals2 = poisson.rvs(request_mean[1])
    returns1 = poisson.rvs(return_mean[0])
    returns2 = poisson.rvs(return_mean[1])
    
    
    available_bikes1 = new_x1 + returns1
    available_bikes2 = new_x2 + returns2
    

    rented1 = min(available_bikes1, rentals1)
    rented2 = min(available_bikes2, rentals2)
    
    
    reward = RENTAL_REWARD * (rented1 + rented2) - MOVE_COST * abs(net_move)
    
    
    new_state1 = max(0, available_bikes1 - rented1)
    new_state2 = max(0, available_bikes2 - rented2)
    
    new_state = (new_state1, new_state2)
    
    return reward, new_state


initial_state = (10, 10)  
action = 2  
reward, new_state = calculate_reward_and_transition(initial_state, action)

print(f"Reward: {reward}, New State: {new_state}")