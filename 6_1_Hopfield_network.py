import numpy as np
import matplotlib.pyplot as plt

# Define functions
def hebbian_learning(patterns):
    """
    Compute the weight matrix using Hebbian learning rule.
    :param patterns: List of binary patterns to store (-1 and 1)
    :return: Weight matrix
    """
    n_neurons = patterns[0].size
    W = np.zeros((n_neurons, n_neurons))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)  # Ensure no self-loops
    return W / len(patterns)  # Normalize by number of patterns

def recall(W, pattern, max_iterations=100):
    """
    Recall a pattern using the Hopfield network.
    :param W: Weight matrix
    :param pattern: Input pattern
    :param max_iterations: Maximum number of update iterations
    :return: Recalled pattern
    """
    n_neurons = pattern.size
    recalled_pattern = pattern.copy()
    for _ in range(max_iterations):
        for i in range(n_neurons):  # Update asynchronously
            recalled_pattern[i] = 1 if np.dot(W[i], recalled_pattern) >= 0 else -1
    return recalled_pattern

def test_stability(W, patterns):
    """
    Test if stored patterns are stable.
    :param W: Weight matrix
    :param patterns: List of stored patterns
    :return: Number of stable patterns
    """
    stable_count = 0
    for p in patterns:
        recalled = recall(W, p)
        if np.array_equal(recalled, p):
            stable_count += 1
    return stable_count

# Generate random 10x10 binary patterns (-1 and 1)
num_neurons = 10 * 10
num_patterns = 10  # Number of patterns to store
patterns = [np.random.choice([-1, 1], num_neurons) for _ in range(num_patterns)]

# Train the Hopfield network
W = hebbian_learning(patterns)

# Test the recall and stability
stable_patterns = test_stability(W, patterns)
capacity = num_patterns / num_neurons  # Calculate capacity

# Display results
print(f"Number of neurons: {num_neurons}")
print(f"Number of patterns stored: {num_patterns}")
print(f"Stable patterns: {stable_patterns}")
print(f"Capacity of Hopfield network: {capacity:.2f}")

# Test pattern recall with a noisy input
test_pattern = patterns[0].copy()
noisy_pattern = test_pattern.copy()
noise_indices = np.random.choice(num_neurons, size=num_neurons // 5, replace=False)
noisy_pattern[noise_indices] *= -1  # Flip some bits

recalled_pattern = recall(W, noisy_pattern)

# Reshape and visualize original, noisy, and recalled patterns
original = test_pattern.reshape((10, 10))
noisy = noisy_pattern.reshape((10, 10))
recalled = recalled_pattern.reshape((10, 10))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(original, cmap='gray')
axes[0].set_title("Original Pattern")
axes[1].imshow(noisy, cmap='gray')
axes[1].set_title("Noisy Pattern")
axes[2].imshow(recalled, cmap='gray')
axes[2].set_title("Recalled Pattern")
plt.show()
