import numpy as np
import random
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, num_arms=10, std_dev=0.01):
        """
        Initialize a non-stationary bandit with the given number of arms.
        :param num_arms: Number of arms (10 by default)
        :param std_dev: Standard deviation for random walk increments (default 0.01)
        """
        self.num_arms = num_arms
        self.std_dev = std_dev
        self.mean_rewards = np.zeros(num_arms)  # Start with all mean rewards equal

    def pull(self, action):
        """
        Pull a bandit arm and get the reward.
        :param action: Index of the arm to pull (0 to num_arms-1)
        :return: Reward (stochastic around the mean reward of the selected arm)
        """
        reward = np.random.normal(self.mean_rewards[action], 1.0)
        return reward

    def random_walk(self):
        """
        Apply a random walk to the mean rewards.
        """
        increments = np.random.normal(0, self.std_dev, self.num_arms)
        self.mean_rewards += increments

def modified_epsilon_greedy(bandit, num_steps=10000, epsilon=0.1, alpha=0.1):
    """
    Modified epsilon-greedy algorithm for the non-stationary bandit.
    :param bandit: NonStationaryBandit instance
    :param num_steps: Number of steps to play
    :param epsilon: Exploration probability
    :param alpha: Step size for exponential reward updates
    :return: Rewards over time, action counts, and action history
    """
    num_arms = bandit.num_arms
    estimated_rewards = np.zeros(num_arms)  # Estimated rewards for each arm
    action_counts = np.zeros(num_arms)  # Number of times each arm is selected
    rewards = []  # Rewards collected at each time step
    action_history = []  # Actions chosen at each time step

    for step in range(num_steps):
        # Decide action (explore or exploit)
        if random.random() < epsilon:  # Explore
            action = random.randint(0, num_arms - 1)
        else:  # Exploit
            action = np.argmax(estimated_rewards)

        # Pull the chosen arm
        reward = bandit.pull(action)
        rewards.append(reward)
        action_history.append(action)

        # Update estimated reward for the chosen action using exponential weighting
        estimated_rewards[action] = (1 - alpha) * estimated_rewards[action] + alpha * reward

        # Apply random walk to the bandit's mean rewards
        bandit.random_walk()

    return rewards, action_counts, action_history

# Initialize the 10-armed non-stationary bandit
bandit = NonStationaryBandit(num_arms=10, std_dev=0.01)

# Run the modified epsilon-greedy algorithm
num_steps = 10000
epsilon = 0.1
alpha = 0.1
rewards, action_counts, action_history = modified_epsilon_greedy(bandit, num_steps, epsilon, alpha)

# Plot cumulative rewards over time
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(rewards), label="Cumulative Reward")
plt.title("Cumulative Reward Over Time (Modified Epsilon-Greedy)")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid()
plt.show()

# Plot action counts
plt.figure(figsize=(12, 6))
plt.bar(range(10), action_counts)
plt.title("Number of Times Each Arm Was Selected")
plt.xlabel("Arm")
plt.ylabel("Count")
plt.grid()
plt.show()

# Display final results
print("Final Estimated Rewards:", np.round(estimated_rewards, 2))
print("Action Counts:", action_counts)
