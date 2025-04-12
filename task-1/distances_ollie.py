import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

#########
# NUMPY #
#########


def distance_cosine_NUMPY(X, Y):
    return 1 - np.dot(Y, X.T).squeeze() / (np.linalg.norm(X) * np.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_NUMPY(X, Y):
    return np.linalg.norm(Y[:, None, :] - X[None, ...], axis=1)


def distance_dot_NUMPY(X, Y):
    return 1 - np.dot(Y[:, None, :], X[None, ...].T).squeeze()


def distance_manhattan_NUMPY(X, Y):
    return np.sum(np.abs(Y[:, None, :] - X[None, ...]), axis=1)


########
# CUPY #
########


def distance_cosine_CUPY(X, Y):
    return 1 - cp.dot(Y, X.T).squeeze() / (cp.linalg.norm(X) * cp.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_CUPY(X, Y):
    return cp.linalg.norm(Y[:, None, :] - X[None, ...], axis=1)


def distance_dot_CUPY(X, Y):
    return 1 - cp.dot(Y[:, None, :], X[None, ...].T).squeeze()


def distance_manhattan_CUPY(X, Y):
    return cp.sum(cp.abs(Y[:, None, :] - X[None, ...]), axis=1)


#############
# BENCHMARK #
#############


def benchmark_and_plot(vector_size=512, batch_size=10000):
    print(
        f"Benchmarking with vector size {vector_size} and batch size {batch_size}\n")

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

    timings_np = {}
    for name, func in functions_np.items():
        start = time.perf_counter()
        _ = func(X_np, Y_np)
        end = time.perf_counter()
        timings_np[name] = end - start

    timings_cp = {}
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

    fig, ax = plt.subplots()
    bar_np = ax.bar(x - width / 2, [timings_np[k]
                    for k in labels], width, label='NumPy')

    bar_cp = ax.bar(x + width / 2, [timings_cp[k]
                                    for k in labels], width, label='CuPy')

    ax.set_ylabel('Time (seconds)')
    ax.set_title(
        f'Distance Function Timings\nVector size: {vector_size}, Batch size: {batch_size}')
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
                        ha='center', va='bottom')

    autolabel(bar_np)
    if bar_cp:
        autolabel(bar_cp)

    plt.tight_layout()
    plt.show()

benchmark_and_plot(2, 1000)
benchmark_and_plot(2 ** 15, 1000)
