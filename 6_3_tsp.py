import numpy as np

def tsp_weights_and_biases(distances):
    """
    Create weights and biases for TSP using Hopfield network.
    :param distances: Distance matrix between cities
    :return: Weight matrix and bias vector
    """
    N = distances.shape[0]  # Number of cities
    num_neurons = N * N
    W = np.zeros((num_neurons, num_neurons))
    theta = np.zeros(num_neurons)

    # Constraints to enforce valid tours
    A = 500  # Penalty for violating constraints
    for i in range(N):
        for j in range(N):
            idx_ij = i * N + j
            # Row constraints: Each city appears in one position
            for k in range(N):
                if j != k:
                    W[idx_ij, i * N + k] -= A
            # Column constraints: Each position is occupied by one city
            for l in range(N):
                if i != l:
                    W[idx_ij, l * N + j] -= A

            # Distance-based term for tour cost
            theta[idx_ij] = -A * N  # Bias term to activate neurons

            for k in range(N):
                if i != k:
                    for l in range(N):
                        idx_kl = k * N + l
                        if j == (l + 1) % N:  # Consecutive cities in the tour
                            W[idx_ij, idx_kl] -= distances[i, k]

    return W, theta

def solve_tsp(W, theta, max_iterations=100):
    """
    Solve TSP using Hopfield network dynamics.
    :param W: Weight matrix
    :param theta: Bias vector
    :param max_iterations: Maximum number of update iterations
    :return: Solution matrix (N x N)
    """
    N = int(np.sqrt(len(theta)))  # Number of cities
    x = np.random.choice([0, 1], len(theta))  # Random initial state

    for _ in range(max_iterations):
        for i in range(len(theta)):
            h = np.dot(W[i], x) + theta[i]
            x[i] = 1 if h > 0 else 0

        # Enforce hard constraints for one-hot encoding
        x = x.reshape((N, N))
        for row in range(N):
            if np.sum(x[row, :]) > 1:
                x[row, :] = np.zeros(N)
                x[row, np.random.randint(0, N)] = 1
        for col in range(N):
            if np.sum(x[:, col]) > 1:
                x[:, col] = np.zeros(N)
                x[np.random.randint(0, N), col] = 1
        x = x.flatten()

    return x.reshape((N, N))

# Generate random distance matrix for 10 cities
N = 10
distances = np.random.randint(1, 100, (N, N))
distances = (distances + distances.T) / 2  # Symmetric matrix

# Solve TSP using Hopfield network
W, theta = tsp_weights_and_biases(distances)
solution = solve_tsp(W, theta)

# Decode solution
tour = np.argmax(solution, axis=1)
total_distance = sum(distances[tour[i], tour[(i + 1) % N]] for i in range(N))

# Display results
print("TSP Solution Tour (City Order):")
print(tour)
print(f"Total Distance: {total_distance}")
