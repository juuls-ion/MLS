# import torch as tch
# import triton
# import triton.language as tl
import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt

#########
# NUMPY #
#########


def distance_cosine_NUMPY(X, Y):
    X = X[None, ...]
    Y = Y[:,None,:]
    return 1 - np.dot(Y, X.T).squeeze() / (np.linalg.norm(X) * np.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_NUMPY(X, Y):
    return np.linalg.norm(Y[:,None,:] - X[None, ...], axis=1)


def distance_dot_NUMPY(X, Y):
    return 1 - np.dot(Y[:,None,:], X[None, ...].T).squeeze()


def distance_manhattan_NUMPY(X, Y):
    return np.sum(np.abs(Y[:,None,:] - X[None, ...]), axis=1)


########
# CUPY #
########


def distance_cosine_CUPY(X, Y):
    X = X[None, ...]
    Y = Y[:,None,:]
    return 1 - cp.dot(Y, X.T).squeeze() / (cp.linalg.norm(X) * cp.linalg.norm(Y, axis=1) + 1e-10)


def distance_l2_CUPY(X, Y):
    return cp.linalg.norm(Y[:,None,:] - X[None, ...], axis=1)


def distance_dot_CUPY(X, Y):
    return 1 - cp.dot(Y[:,None,:], X[None, ...].T).squeeze()


def distance_manhattan_CUPY(X, Y):
    return cp.sum(cp.abs(Y[:,None,:] - X[None, ...]), axis=1)


#########
# TORCH #
#########

"""
def distance_cosine_TORCH(X, Y):
    return 1 - (tch.matmul(X, Y.T) / (tch.norm(X, dim=1, keepdim=True) * tch.norm(Y, dim=1, keepdim=True).T))


def distance_l2_TORCH(X, Y):
    return tch.sum(tch.square(X[:, None, :] - Y[None, :, :]), dim=-1)


def distance_dot_TORCH(X, Y):
    return 1 - tch.matmul(X, Y.T)


def distance_manhattan_TORCH(X, Y):
    return tch.sum(tch.abs(X[:, None, :] - Y[None, :, :]), dim=-1)
    """


##########
# TRITON #
##########

"""
# ---------- Cosine Distance ----------
@triton.jit
def cosine_kernel(X_ptr, Y_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + row * D + offs)
    y = tl.load(Y_ptr + col * D + offs)

    dot = tl.sum(x * y)
    x_norm = tl.sqrt(tl.sum(x * x))
    y_norm = tl.sqrt(tl.sum(y * y))
    cosine_sim = dot / (x_norm * y_norm)
    tl.store(out_ptr + row * N + col, 1 - cosine_sim)


def distance_cosine_TRITON(X, Y):
    N, D = X.shape
    out = tch.empty((N, N), device='cuda', dtype=X.dtype)
    grid = (N, N)
    cosine_kernel[grid](X_ptr=X, Y_ptr=Y, out_ptr=out, N=N, D=D, BLOCK_SIZE=D)
    return out


# ---------- L2 Distance ----------
@triton.jit
def l2_kernel(X_ptr, Y_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + row * D + offs)
    y = tl.load(Y_ptr + col * D + offs)

    diff = x - y
    dist = tl.sum(diff * diff)
    tl.store(out_ptr + row * N + col, dist)


def distance_l2_TRITON(X, Y):
    N, D = X.shape
    out = tch.empty((N, N), device='cuda', dtype=X.dtype)
    grid = (N, N)
    l2_kernel[grid](X_ptr=X, Y_ptr=Y, out_ptr=out, N=N, D=D, BLOCK_SIZE=D)
    return out


# ---------- Dot Distance ----------
@triton.jit
def dot_kernel(X_ptr, Y_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + row * D + offs)
    y = tl.load(Y_ptr + col * D + offs)

    dot = tl.sum(x * y)
    tl.store(out_ptr + row * N + col, 1 - dot)


def distance_dot_TRITON(X, Y):
    N, D = X.shape
    out = tch.empty((N, N), device='cuda', dtype=X.dtype)
    grid = (N, N)
    dot_kernel[grid](X_ptr=X, Y_ptr=Y, out_ptr=out, N=N, D=D, BLOCK_SIZE=D)
    return out


# ---------- Manhattan Distance ----------
@triton.jit
def manhattan_kernel(X_ptr, Y_ptr, out_ptr, N, D, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + row * D + offs)
    y = tl.load(Y_ptr + col * D + offs)

    dist = tl.sum(tl.abs(x - y))
    tl.store(out_ptr + row * N + col, dist)


def distance_manhattan_TRITON(X, Y):
    N, D = X.shape
    out = tch.empty((N, N), device='cuda', dtype=X.dtype)
    grid = (N, N)
    manhattan_kernel[grid](X_ptr=X, Y_ptr=Y, out_ptr=out,
                           N=N, D=D, BLOCK_SIZE=D)
    return out
"""


###########
# TESTING #
###########

"""
def test_function(fn, d, c):
    times = []

    lib = fn.__name__.split("_")[2]
    func = fn.__name__.split("_")[1]
    print(f"{lib}: {func}")

    xs = np.random.rand(c, d)
    ys = np.random.rand(c, d)
    if lib == "CUPY":
        xs = cp.array(xs)
        ys = cp.array(ys)

    start = time.time()
    for x, y in zip(xs, ys):
        fn(x, y)
        times.append(time.time() - start)

    return times


# 2D vectors
print("2D Vectors")
np_cosine_2d = test_function(distance_cosine_NUMPY, 2, 1_000)
cp_cosine_2d = test_function(distance_cosine_CUPY, 2, 1_000)
np_l2_2d = test_function(distance_l2_NUMPY, 2, 1_000)
cp_l2_2d = test_function(distance_l2_CUPY, 2, 1_000)
np_dot_2d = test_function(distance_dot_NUMPY, 2, 1_000)
cp_dot_2d = test_function(distance_dot_CUPY, 2, 1_000)
np_manhattan_2d = test_function(distance_manhattan_NUMPY, 2, 1_000)
cp_manhattan_2d = test_function(distance_manhattan_CUPY, 2, 1_000)

# 2^15D vectors
print("2^15D Vectors")
np_cosine_2_15d = test_function(distance_cosine_NUMPY, 2 ** 15, 1_000)
cp_cosine_2_15d = test_function(distance_cosine_CUPY, 2 ** 15, 1_000)
np_l2_2_15d = test_function(distance_l2_NUMPY, 2 ** 15, 1_000)
cp_l2_2_15d = test_function(distance_l2_CUPY, 2 ** 15, 1_000)
np_dot_2_15d = test_function(distance_dot_NUMPY, 2 ** 15, 1_000)
cp_dot_2_15d = test_function(distance_dot_CUPY, 2 ** 15, 1_000)
np_manhattan_2_15d = test_function(distance_manhattan_NUMPY, 2 ** 15, 1_000)
cp_manhattan_2_15d = test_function(distance_manhattan_CUPY, 2 ** 15, 1_000)

# Plot graph
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

ax1 = axes[0]
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("2D Calculations")
ax1.plot(np_cosine_2d, range(len(np_cosine_2d)), label="np_cosine_2d")
ax1.plot(cp_cosine_2d, range(len(cp_cosine_2d)), label="cp_cosine_2d")
ax1.plot(np_l2_2d, range(len(np_l2_2d)), label="np_l2_2d")
ax1.plot(cp_l2_2d, range(len(cp_l2_2d)), label="cp_l2_2d")
ax1.plot(np_dot_2d, range(len(np_dot_2d)), label="np_dot_2d")
ax1.plot(cp_dot_2d, range(len(cp_dot_2d)), label="cp_dot_2d")
ax1.plot(np_manhattan_2d, range(len(np_manhattan_2d)), label="np_manhattan_2d")
ax1.plot(cp_manhattan_2d, range(len(cp_manhattan_2d)), label="cp_manhattan_2d")
ax1.legend()

ax2 = axes[1]
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("2^15D Calculations")
ax2.plot(np_cosine_2_15d, range(len(np_cosine_2_15d)), label="np_cosine_2_15d")
ax2.plot(cp_cosine_2_15d, range(len(cp_cosine_2_15d)), label="cp_cosine_2_15d")
ax2.plot(np_l2_2_15d, range(len(np_l2_2_15d)), label="np_l2_2_15d")
ax2.plot(cp_l2_2_15d, range(len(cp_l2_2_15d)), label="cp_l2_2_15d")
ax2.plot(np_dot_2_15d, range(len(np_dot_2_15d)), label="np_dot_2_15d")
ax2.plot(cp_dot_2_15d, range(len(cp_dot_2_15d)), label="cp_dot_2_15d")
ax2.plot(np_manhattan_2_15d, range(len(np_manhattan_2_15d)), label="np_manhattan_2_15d")
ax2.plot(cp_manhattan_2_15d, range(len(cp_manhattan_2_15d)), label="cp_manhattan_2_15d")
ax2.legend()

plt.savefig(f"distance_functions.png")
"""

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
