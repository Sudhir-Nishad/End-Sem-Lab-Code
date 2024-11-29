import numpy as np
import random

class BinaryBandit:
    def __init__(self, success_prob):
        """
        Initialize a binary bandit with a given success probability.
        :param success_prob: Probability of returning a reward of 1
        """
        self.success_prob = success_prob

    def pull(self):
        """
        Simulate pulling the bandit arm.
        :return: 1 (success) or 0 (failure)
        """
        return 1 if random.random() < self.success_prob else 0

def epsilon_greedy(bandits, num_steps=1000, epsilon=0.1):
    """
    Epsilon-greedy algorithm to maximize expected reward for binary bandits.
    :param bandits: List of BinaryBandit objects
    :param num_steps: Number of steps to play
    :param epsilon: Exploration probability
    :return: Estimated rewards, total rewards, and action history
    """
    n = len(bandits)
    estimated_rewards = np.zeros(n)  # Estimated reward for each bandit
    action_counts = np.zeros(n)  # Number of times each bandit is selected
    total_reward = 0
    action_history = []

    for step in range(num_steps):
        # Decide action (explore or exploit)
        if random.random() < epsilon:  # Explore
            action = random.randint(0, n - 1)
        else:  # Exploit
            action = np.argmax(estimated_rewards)

        # Pull the selected bandit's arm
        reward = bandits[action].pull()
        total_reward += reward
        action_history.append(action)

        # Update estimated reward for the chosen action
        action_counts[action] += 1
        estimated_rewards[action] += (reward - estimated_rewards[action]) / action_counts[action]

    return estimated_rewards, total_reward, action_history

# Simulate two binary bandits
banditA = BinaryBandit(success_prob=0.7)  # Bandit A has 70% success rate
banditB = BinaryBandit(success_prob=0.5)  # Bandit B has 50% success rate

# List of bandits
bandits = [banditA, banditB]

# Run epsilon-greedy algorithm
epsilon = 0.1
num_steps = 1000
estimated_rewards, total_reward, action_history = epsilon_greedy(bandits, num_steps, epsilon)

# Display results
print("Estimated Rewards:", estimated_rewards)
print("Total Reward:", total_reward)
print("Action History (last 20 actions):", action_history[-20:])
