import cupy as cp
from tqdm import tqdm
import numpy as np
import subprocess
import matplotlib.pyplot as plt

def gpu_stats():
    query_metrics = ["utilization.gpu"]
    metrics_str = ",".join(query_metrics)
    command = [
        'nvidia-smi',
        f'--query-gpu={metrics_str}',
        '--format=csv,noheader,nounits',
        f'--id=0'
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()

gpu_utils = []

def overall_knn(q_gpu, p_cpu, k, batch_size=8192):
    n_q = q_gpu.shape[0]
    n_p = p_cpu.shape[0]
    global gpu_utils

    # Preallocate arrays
    overall_top_k_distance = cp.full((n_q, k), cp.inf)
    overall_top_k_indices = cp.full((n_q, k), -1)

    for start_idx in range(0, n_p, batch_size):
        end_idx = min(start_idx + batch_size, n_p)
        p_batch = cp.asarray(p_cpu[start_idx:end_idx])
        diff = cp.sum((q_gpu[:, None, :] - p_batch[None, ...]) ** 2, axis=2)
        batch_distances = cp.sqrt(diff)

        batch_indices_abs = cp.arange(start_idx, end_idx)
        batch_indices_expanded = cp.tile(batch_indices_abs[None, :], (n_q, 1))

        concatenated_distances = cp.concatenate((overall_top_k_distance, batch_distances), axis=1)
        concatenated_indices = cp.concatenate((overall_top_k_indices, batch_indices_expanded), axis=1)

        if not (start_idx % 100):
            gpu_utils.append(gpu_stats())

        top_k_indices = cp.argpartition(concatenated_distances, k, axis=1)[:, :k]
        overall_top_k_distance = cp.take_along_axis(concatenated_distances, top_k_indices, axis=1)
        overall_top_k_indices = cp.take_along_axis(concatenated_indices, top_k_indices, axis=1)

    return overall_top_k_distance, overall_top_k_indices

def refine_clusters(p_gpu, current_label_index):
    """
    Compute new centroids by averaging the points in each cluster.
    """
    n_clusters = int(cp.max(current_label_index)) + 1
    n_features = p_gpu.shape[1]
    new_centroids = cp.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        mask = current_label_index == i
        cluster_points = p_gpu[mask]
        if cluster_points.shape[0] > 0:
            new_centroids[i] = cp.mean(cluster_points, axis=0)
    return new_centroids


#@cp.fuse('knn')
def knn(q_cpu, p_cpu, k, batch_size=10_000_000):
    n_q = q_cpu.shape[0]
    q_indices = []
    q_distances = []
    for start_idx in range(0, n_q, batch_size):
        end_idx = min(start_idx + batch_size, n_q)
        q_gpu = cp.asarray(q_cpu[start_idx:end_idx], dtype=cp.float16)
        distances, indices = overall_knn(q_gpu, p_cpu, k, batch_size)
        q_indices.append(indices)
        q_distances.append(distances)
    return cp.concatenate(q_indices, axis=0), cp.concatenate(q_distances, axis=0)


gpu_utils = []
def get_clusters(p_cpu, c, max_iter=10):
    global gpu_utils
    n_p = p_cpu.shape[0]
    centroids = p_cpu[np.random.randint(0, n_p, (c,))]
    indices, distances = knn(p_cpu, centroids, 1)
    indices = indices.flatten()
    p_gpu = cp.asarray(p_cpu)

    for i in tqdm(range(max_iter)):
        new_means = refine_clusters(p_gpu, indices)
        new_indices, new_distances = knn(p_cpu, cp.asnumpy(new_means), 1)
        new_indices = new_indices.flatten()
        gpu_utils.append(gpu_stats())

        if cp.all(new_indices == indices):
            print("Converged.")
            break

        indices = new_indices
        distances = new_distances


# ---- RUN ----

N = 1_00_000_000
D = 2
K = 5
Q = 666

print("Allocating random data...")
points_cpu = np.random.rand(N, D)
print("Running KMeans...")
get_clusters(points_cpu, c=3)



print("Plotting GPU utilization stats...")
plt.plot(list(map(int, gpu_utils)))
plt.ylabel("GPU Utilization (%)")
plt.title("GPU Utilization During KNN")
plt.savefig("gpu_util.png")
print("Saved GPU utilization plot as gpu_util.png")
