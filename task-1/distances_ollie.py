import torch as tch
import cupy as cp
import triton as tr

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


def distance_cosine_TRITON(X, Y, out):
    row, col = tr.program_id(0), tr.program_id(1)
    x = X[row, :]
    y = Y[col, :]
    out[row, col] = 1 - tr.sum(x * y) / (tr.sqrt(tr.sum(x * x)) * tr.sqrt(tr.sum(y * y)))
    return out


def distance_l2_TRITON(X, Y, out):
    row, col = tr.program_id(0), tr.program_id(1)
    out[row, col] = tr.sum(tr.square(X[row, :] - Y[col, :]))
    return out


def distance_dot_TRITON(X, Y, out):
    row, col = tr.program_id(0), tr.program_id(1)
    out[row, col] = 1 - tr.sum(X[row, :] * Y[col, :])
    return out


def distance_manhattan_TRITON(X, Y, out):
    row, col = tr.program_id(0), tr.program_id(1)
    out[row, col] = tr.sum(tr.abs(X[row, :] - Y[col, :]))
    return out


###########
# TESTING #
###########


def generate_vectors(n, d):
    """
    Generates n random vectors of dimension d.
    Converts them to cupy, torch, and triton tensors.
    """
    torch_vectors = tch.randn(n, d)
    cupy_vectors = cp.asarray(torch_vectors.numpy())
    triton_vectors = tr.from_numpy(torch_vectors.numpy())
    return torch_vectors, cupy_vectors, triton_vectors


if __name__ == "__main__":
    tch_vs, cp_vs, tr_vs = generate_vectors(10, 2)
    print("Torch vectors:\n", tch_vs)
    print("CuPy vectors:\n", cp_vs)
    print("Triton vectors:\n", tr_vs)