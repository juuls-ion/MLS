import torch as tch
import cupy as cp
import triton
import triton.language as tl

########
# CUPY #
########


def distance_cosine_CUPY(X, Y):
    return 1 - (cp.dot(X, Y.T) / (cp.linalg.norm(X, axis=1, keepdims=True) * cp.linalg.norm(Y, axis=1, keepdims=True)))


def distance_l2_CUPY(X, Y):
    return cp.sum(cp.square(X[:, None, :] - Y[None, :, :]), axis=-1)


def distance_dot_CUPY(X, Y):
    return 1 - cp.dot(X, Y.T)


def distance_manhattan_CUPY(X, Y):
    return cp.sum(cp.abs(X[:, None, :] - Y[None, :, :]), axis=-1)


#########
# TORCH #
#########


def distance_cosine_TORCH(X, Y):
    return 1 - (tch.matmul(X, Y.T) / (tch.norm(X, dim=1, keepdim=True) * tch.norm(Y, dim=1, keepdim=True).T))


def distance_l2_TORCH(X, Y):
    return tch.sum(tch.square(X[:, None, :] - Y[None, :, :]), dim=-1)


def distance_dot_TORCH(X, Y):
    return 1 - tch.matmul(X, Y.T)


def distance_manhattan_TORCH(X, Y):
    return tch.sum(tch.abs(X[:, None, :] - Y[None, :, :]), dim=-1)


##########
# TRITON #
##########

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
    manhattan_kernel[grid](X_ptr=X, Y_ptr=Y, out_ptr=out, N=N, D=D, BLOCK_SIZE=D)
    return out


###########
# TESTING #
###########


def generate_vectors(n, d):
    """
    Generates n random vectors of dimension d.
    Converts them to CUDA tensors for PyTorch and CuPy.
    Triton uses the same PyTorch CUDA tensors.
    """
    torch_vectors = tch.randn(n, d, device="cuda")
    cupy_vectors = cp.asarray(torch_vectors.cpu().numpy())
    return torch_vectors, cupy_vectors


if __name__ == "__main__":
    tch_vs, cp_vs = generate_vectors(10, 2)
    print("Torch vectors:\n", tch_vs)
    print("CuPy vectors:\n", cp_vs)
