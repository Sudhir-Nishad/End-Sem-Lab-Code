import numpy as np

def eight_rook_weights(n=8):
    """
    Create weights and biases for the Eight-rook problem.
    :param n: Size of the chessboard (n x n)
    :return: Weight matrix (W) and bias vector (theta)
    """
    num_neurons = n * n
    W = np.zeros((num_neurons, num_neurons))
    theta = np.ones(num_neurons)  # Encourage placing rooks

    for i in range(n):
        for j in range(n):
            idx_ij = i * n + j
            for k in range(n):
                for l in range(n):
                    idx_kl = k * n + l
                    if idx_ij != idx_kl:
                        # Penalize same row
                        if i == k and j != l:
                            W[idx_ij, idx_kl] = -2
                        # Penalize same column
                        if j == l and i != k:
                            W[idx_ij, idx_kl] = -2

    return W, theta

def solve_eight_rook(W, theta, max_iterations=100):
    """
    Solve the Eight-rook problem using Hopfield dynamics.
    :param W: Weight matrix
    :param theta: Bias vector
    :param max_iterations: Maximum number of update iterations
    :return: Solution matrix (n x n)
    """
    n = int(np.sqrt(len(theta)))  # Board size
    x = np.random.choice([0, 1], len(theta))  # Random initial state

    for _ in range(max_iterations):
        for i in range(len(theta)):
            h = np.dot(W[i], x) + theta[i]
            x[i] = 1 if h > 0 else 0

        # Enforce one rook per row and column
        x = x.reshape((n, n))
        for row in range(n):
            if np.sum(x[row, :]) > 1:
                x[row, :] = np.zeros(n)
                x[row, np.random.randint(0, n)] = 1
        for col in range(n):
            if np.sum(x[:, col]) > 1:
                x[:, col] = np.zeros(n)
                x[np.random.randint(0, n), col] = 1
        x = x.flatten()

    return x.reshape((n, n))

# Solve the Eight-rook problem
n = 8
W, theta = eight_rook_weights(n)
solution = solve_eight_rook(W, theta)

# Display solution
print("Eight-rook Solution:")
print(solution)

# Verify solution
for row in range(n):
    assert np.sum(solution[row, :]) == 1, "Row constraint violated"
for col in range(n):
    assert np.sum(solution[:, col]) == 1, "Column constraint violated"

print("Solution is valid!")
