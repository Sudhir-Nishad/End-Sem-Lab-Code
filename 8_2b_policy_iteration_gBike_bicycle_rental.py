import numpy as np
import matplotlib.pyplot as plt

class GbikeProblem:
    def __init__(self):
        # Problem parameters
        self.max_bikes = 20  # Maximum bikes at each location
        self.max_move = 5    # Maximum bikes that can be moved between locations per night
        
        # Rental and moving costs
        self.rental_reward = 10  # Reward per bike rented
        self.move_cost = 2       # Cost per bike moved (except first free move)
        self.parking_overflow_cost = 4  # Cost for second parking lot if > 10 bikes
        
        # Rental and return probabilities (Poisson distribution parameters)
        self.rental_lambda_1 = 3  # Average rentals at location 1
        self.rental_lambda_2 = 2  # Average rentals at location 2
        self.return_lambda_1 = 3  # Average returns at location 1
        self.return_lambda_2 = 2  # Average returns at location 2
        
        # Policy and value initialization
        self.policy = np.zeros((self.max_bikes + 1, self.max_bikes + 1), dtype=int)
        self.value = np.zeros((self.max_bikes + 1, self.max_bikes + 1))
        
    def poisson_probability(self, n, lam):
        """Calculate Poisson probability for n events with mean lambda."""
        return (lam ** n * np.exp(-lam)) / np.math.factorial(n)
    
    def expected_rentals_and_returns(self, n, rental_lambda, return_lambda):
        """Calculate expected rentals and returns."""
        expected_reward = 0
        expected_remaining = 0
        
        for r in range(n + 1):
            for rented in range(r + 1):
                # Probability of this scenario
                prob = (self.poisson_probability(r, rental_lambda) * 
                        self.poisson_probability(rented, r))
                
                # Actual rentals (limited by available bikes)
                actual_rented = min(rented, n)
                
                # Reward from rentals
                reward = actual_rented * self.rental_reward
                
                # Remaining bikes after rentals
                remaining = n - actual_rented
                
                # Add to expected values
                expected_reward += prob * reward
                expected_remaining += prob * remaining
        
        return expected_reward, expected_remaining
    
    def compute_state_value(self, state_bikes_1, state_bikes_2, policy_move):
        """Compute value of a state based on current policy."""
        # Apply policy move (from location 1 to 2 if positive, 2 to 1 if negative)
        if policy_move > 0:
            # First move is free
            if policy_move > 1:
                state_move_cost = (policy_move - 1) * self.move_cost
            else:
                state_move_cost = 0
            state_bikes_1 -= policy_move
            state_bikes_2 += policy_move
        else:
            # First move is free
            if abs(policy_move) > 1:
                state_move_cost = (abs(policy_move) - 1) * self.move_cost
            else:
                state_move_cost = 0
            state_bikes_1 += abs(policy_move)
            state_bikes_2 -= abs(policy_move)
        
        # Parking overflow cost
        parking_cost = 0
        if state_bikes_1 > 10 or state_bikes_2 > 10:
            parking_cost = self.parking_overflow_cost
        
        # Expected rentals and returns for each location
        reward_1, remaining_1 = self.expected_rentals_and_returns(
            state_bikes_1, self.rental_lambda_1, self.return_lambda_1)
        reward_2, remaining_2 = self.expected_rentals_and_returns(
            state_bikes_2, self.rental_lambda_2, self.return_lambda_2)
        
        # Total expected value
        total_reward = reward_1 + reward_2 - state_move_cost - parking_cost
        
        return total_reward, remaining_1, remaining_2
    
    def policy_evaluation(self):
        """Evaluate current policy."""
        while True:
            delta = 0
            for i in range(self.max_bikes + 1):
                for j in range(self.max_bikes + 1):
                    old_value = self.value[i][j]
                    move = self.policy[i][j]
                    
                    # Compute value for current state
                    total_reward, remaining_1, remaining_2 = self.compute_state_value(i, j, move)
                    self.value[i][j] = total_reward
                    
                    # Update delta for convergence check
                    delta = max(delta, abs(old_value - self.value[i][j]))
            
            # Convergence threshold
            if delta < 0.1:
                break
    
    def policy_improvement(self):
        """Improve policy based on current value function."""
        policy_stable = True
        
        for i in range(self.max_bikes + 1):
            for j in range(self.max_bikes + 1):
                old_action = self.policy[i][j]
                
                # Try all possible moves
                best_value = float('-inf')
                best_move = 0
                
                for move in range(-self.max_move, self.max_move + 1):
                    # Check if move is valid
                    if (0 <= i - move <= self.max_bikes and 
                        0 <= j + move <= self.max_bikes):
                        
                        # Compute value for this move
                        total_reward, _, _ = self.compute_state_value(i, j, move)
                        
                        # Update best move if better value found
                        if total_reward > best_value:
                            best_value = total_reward
                            best_move = move
                
                # Update policy
                self.policy[i][j] = best_move
                
                # Check if policy is stable
                if old_action != best_move:
                    policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self):
        """Main policy iteration algorithm."""
        iterations = 0
        while True:
            # Policy Evaluation
            self.policy_evaluation()
            
            # Policy Improvement
            policy_stable = self.policy_improvement()
            
            iterations += 1
            print(f"Iteration {iterations}")
            
            # Stop if policy is stable
            if policy_stable:
                break
        
        return iterations
    
    def visualize_policy(self):
        """Visualize the final policy."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.policy, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Bikes to Move')
        plt.title('Optimal Policy for Bike Movement')
        plt.xlabel('Bikes at Location 2')
        plt.ylabel('Bikes at Location 1')
        plt.tight_layout()
        plt.show()

# Run the policy iteration
problem = GbikeProblem()
total_iterations = problem.policy_iteration()
print(f"Policy converged in {total_iterations} iterations")
problem.visualize_policy()