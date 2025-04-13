import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt


###############
# NUMPY (CPU) #
###############

def distance_cosine_NUMPY(X, Y):
    """
    Calculates the cosine distance of X with each row in Y using NumPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return 1 - np.dot(Y, X.T).squeeze() / (np.linalg.norm(X) * np.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_NUMPY(X, Y):
    """
    Calculates the Euclidean (l2) distance of X with each row in Y using NumPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return np.linalg.norm(Y[:, None, :] - X[None, ...], axis=1)


def distance_dot_NUMPY(X, Y):
    """
    Calculates the dot product distance of X with each row in Y using NumPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return 1 - np.dot(Y[:, None, :], X[None, ...].T).squeeze()


def distance_manhattan_NUMPY(X, Y):
    """
    Calculates the Manhattan (l1) distance of X with each row in Y using NumPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return np.sum(np.abs(Y[:, None, :] - X[None, ...]), axis=1)


##############
# CUPY (GPU) #
##############

def distance_cosine_CUPY(X, Y):
    """
    Calculates the cosine distance of X with each row in Y using CuPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return 1 - cp.dot(Y, X.T).squeeze() / (cp.linalg.norm(X) * cp.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_CUPY(X, Y):
    """
    Calculates the Euclidean (l2) distance of X with each row in Y using CuPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return cp.linalg.norm(Y[:, None, :] - X[None, ...], axis=1)


def distance_dot_CUPY(X, Y):
    """
    Calculates the dot product distance of X with each row in Y using CuPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return 1 - cp.dot(Y[:, None, :], X[None, ...].T).squeeze()


def distance_manhattan_CUPY(X, Y):
    """
    Calculates the Manhattan (l1) distance of X with each row in Y using CuPy.

    :param X: A single vector of dimension d.
    :param Y: A matrix of dimension dxn.
    :returns: A matrix of dimension dxn.
    """
    return cp.sum(cp.abs(Y[:, None, :] - X[None, ...]), axis=1)


################
# BENCHMARKING #
################

def benchmark_and_plot(ax, vector_size, batch_size):
    """
    Benchmarks the distance functions for both NumPy and CuPy and plots a graph on the given axis.

    :param ax: The axis to plot the graph on.
    :param vector_size: The dimensionality of the vectors to benchmark on.
    :param batch_size: The number of vectors to benchmark against.
    """
    print(
        f"Benchmarking with vector dimension {vector_size} and batch size {batch_size}\n")

    # Generate random test data
    X_np = np.random.rand(vector_size).astype(np.float32)
    Y_np = np.random.rand(batch_size, vector_size).astype(np.float32)

    X_cp = cp.asarray(X_np)
    Y_cp = cp.asarray(Y_np)

    # Function sets
    functions_np = {
        "Cosine": distance_cosine_NUMPY,
        "L2": distance_l2_NUMPY,
        "Dot": distance_dot_NUMPY,
        "Manhattan": distance_manhattan_NUMPY,
    }

    functions_cp = {
        "Cosine": distance_cosine_CUPY,
        "L2": distance_l2_CUPY,
        "Dot": distance_dot_CUPY,
        "Manhattan": distance_manhattan_CUPY,
    }

    # Keep track of execution times for each function
    timings_np = {}
    for name, func in functions_np.items():
        start = time.perf_counter()
        _ = func(X_np, Y_np)
        end = time.perf_counter()
        timings_np[name] = end - start

    timings_cp = {}
    # Handle GPU synchonisation
    cp.cuda.Device(0).synchronize()
    for name, func in functions_cp.items():
        start = time.perf_counter()
        _ = func(X_cp, Y_cp)
        cp.cuda.Device(0).synchronize()
        end = time.perf_counter()
        timings_cp[name] = end - start

    # Plotting
    labels = list(timings_np.keys())
    x = np.arange(len(labels))
    width = 0.35

    bar_np = ax.bar(x - width / 2, [timings_np[k]
                    for k in labels], width, label="NumPy")
    bar_cp = ax.bar(x + width / 2, [timings_cp[k]
                    for k in labels], width, label="CuPy")

    ax.set_ylabel("Time (seconds)")
    ax.set_title(
        f"Distance Function Timings\nVector dimension: {vector_size}, Batch size: {batch_size}", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom")

    autolabel(bar_np)
    if bar_cp:
        autolabel(bar_cp)

    plt.tight_layout()


def generate_graphs(log=False):
    """
    Generates the graphs for the distance function benchmarks.
    :param log: If True, use logarithmic scale for the y-axis.
    """
    # Create two plots, one for 2D vectors, one for 2^15D vectors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    # Remove lines from top and right of graphs
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Option for logarithmic or linear scale
    if log:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    # Run the benchmarking
    benchmark_and_plot(ax1, 2, 4_000_000)
    benchmark_and_plot(ax2, 2 ** 15, 10_000)

    # Save the file with the appropriate name
    plt.savefig(f"distance_benchmarks{"_log" if log else ""}.png")


if __name__ == "__main__":
    # Generate the graphs
    generate_graphs(log=False)
