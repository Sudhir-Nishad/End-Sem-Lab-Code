import numpy as np

def value_iteration(grid, actions, P, R, gamma=0.9, theta=1e-6):
 
   
    V = {s: 0 for s in grid}
    
    while True:
        delta = 0  
        
        
        V_prime = V.copy()
        
        for s in grid:
            
            if s in R and R[s] is None:
                continue
            
            max_value = float('-inf')  
            
           
            for a in actions:
                expected_value = 0
                
                
                for s_prime, prob in P[s][a].items():
                    expected_value += prob * V[s_prime]
                
               
                total_value = R[s] + gamma * expected_value
                max_value = max(max_value, total_value)
            
            
            V_prime[s] = max_value
            
            delta = max(delta, abs(V_prime[s] - V[s]))
        
        V = V_prime
        
        if delta < theta:
            break
    
    return V

if __name__ == "__main__":
    grid = ["A", "B", "C", "D"]  
    actions = ["left", "right", "up", "down"]  
    
    
    P = {
        "A": {
            "left": {"A": 1.0},
            "right": {"B": 1.0},
            "up": {"A": 1.0},
            "down": {"C": 1.0},
        },
        "B": {
            "left": {"A": 1.0},
            "right": {"B": 1.0},
            "up": {"B": 1.0},
            "down": {"D": 1.0},
        },
        "C": {
            "left": {"C": 1.0},
            "right": {"D": 1.0},
            "up": {"A": 1.0},
            "down": {"C": 1.0},
        },
        "D": {
            "left": {"C": 1.0},
            "right": {"D": 1.0},
            "up": {"B": 1.0},
            "down": {"D": 1.0},
        },
    }
    
    R = {
        "A": 0,
        "B": 1,
        "C": -1,
        "D": 0,
    }
    
    optimal_values = value_iteration(grid, actions, P, R)
    
    
    for state, value in optimal_values.items():
        print(f"State {state}: {value:.2f}")