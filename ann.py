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
    # Determine the ratio of batch size for queries and points
    n_q = q_cpu.shape[0]
    n_p = p_cpu.shape[0]
    
    p_batch_size = int(batch_size / n_q) 
    print(p_batch_size)
    q_batch_size = n_q
    print(q_batch_size)
    indices = np.empty((n_q, k), dtype=np.int32)
    distances = np.empty((n_q, k), dtype=np.float32)

    for start_idx in tqdm(range(0, n_q, q_batch_size)):
        end_idx = min(start_idx + batch_size, n_q)
        q_gpu = cp.asarray(q_cpu[start_idx:end_idx])
        batch_distances, batch_indices = overall_knn(q_gpu, p_cpu, k, p_batch_size)
        indices[start_idx:end_idx] = cp.asnumpy(batch_indices)
        distances[start_idx:end_idx] = cp.asnumpy(batch_distances)
    return indices, distances


def euclidean_distance(a, b):
    return cp.linalg.norm(a - b, axis=-1)

def initialize_centers(data, k):
    idx = np.random.choice(len(data), size=k, replace=False)
    return cp.asarray(data[idx])

def mini_batch_kMeans(data, k, iterations=10, batch_size=20):
    centers = initialize_centers(data, k)

    for _ in range(iterations):
        idx = np.random.choice(len(data), size=batch_size, replace=False)
        batch = cp.asarray(data[idx])

        dists = cp.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
        labels = cp.argmin(dists, axis=1)

        for i in range(k):
            mask = labels == i
            if cp.any(mask):
                centers[i] = batch[mask].mean(axis=0)

    # Final full assignment
    full_data = cp.asarray(data)
    final_dists = cp.linalg.norm(full_data[:, None, :] - centers[None, :, :], axis=2)
    full_labels = cp.argmin(final_dists, axis=1)

    return centers, full_labels.get()
gpu_utils = []
def get_clusters(p_cpu, c, max_iter=10):
    global gpu_utils
    n_p = p_cpu.shape[0]
    centroids = p_cpu[np.random.randint(0, n_p, (c,))]
    indices, distances = knn(p_cpu, centroids, 1)
    indices = indices.flatten()
    p_gpu = cp.asarray(p_cpu)

    for i in tqdm(range(max_iter)):
        centroids = refine_clusters(p_gpu, indices)
        new_indices, new_distances = knn(p_cpu, cp.asnumpy(centroids), 1)
        new_indices = new_indices.flatten()
        gpu_utils.append(gpu_stats())

        indices = new_indices
        distances = new_distances
    return centroids, indices

def group_by_cluster(points, labels):
    sorted_label_indices = cp.argsort(labels)
    sorted_labels = labels[sorted_label_indices]
    sorted_points = points[sorted_label_indices]
    change_indices = cp.where(cp.diff(sorted_labels) != 0)[0] + 1
    start_indices = cp.concatenate((cp.array([0]), change_indices)).get()
    end_indices = cp.concatenate((change_indices, cp.array([len(labels)]))).get()
    return [(sorted_points[start:end], sorted_label_indices[start:end]) for start, end in zip(start_indices, end_indices)]

def ann(p_cpu, q_cpu, k, c=128):
    centroids, labels = mini_batch_kMeans(p_cpu, c, iterations=10)

    grouped_p = group_by_cluster(p_cpu, labels)
    # Now, find the nearest centroid for each query
    q_gpu = cp.asarray(q_cpu)
    indices, _ = knn(q_gpu, centroids, 1)
    indices = indices.flatten()
    # Group the queries by their nearest centroid
    grouped_q = group_by_cluster(q_cpu, indices)
    # Now, find the nearest points in each group
    results = np.empty((q_cpu.shape[0], k), dtype=np.int32)
    for i in range(len(grouped_q)):
        q_group, q_reverse_indices = grouped_q[i]
        p_group, p_reverse_indices = grouped_p[i]
        # Find the nearest points in the group
        indices, _ = knn(q_group, p_group, k)
        # Convert the indices to the original point indices
        indices = p_reverse_indices[indices]
        # Store the results in the original order
        results[q_reverse_indices] = indices
    return results


def calculate_recall(ann_indices, knn_indices, k):
    # Input validation
    if ann_indices.shape != knn_indices.shape:
        raise ValueError(f"Shape mismatch: ANN indices shape {ann_indices.shape} "
                         f"does not match KNN indices shape {knn_indices.shape}")

    if ann_indices.ndim != 2 or ann_indices.shape[1] != k:
        # Derive k if needed, but better to be explicit
        # k = ann_indices.shape[1] if ann_indices.ndim == 2 else 0
        raise ValueError(f"Number of neighbors k ({k}) does not match "
                         f"the second dimension of the index arrays ({ann_indices.shape[1]})")

    num_queries = ann_indices.shape[0]

    if num_queries == 0:
        return 1.0 # Perfect recall if there are no queries to test
    if k == 0:
        return 0.0 # Cannot recall anything if k=0

    total_common_neighbors = 0
    for i in range(num_queries):
        # Convert rows to sets for efficient intersection checking
        ann_set = set(ann_indices[i, :])
        knn_set = set(knn_indices[i, :])

        # Find how many true neighbors were found by ANN
        common_neighbors = ann_set.intersection(knn_set)
        total_common_neighbors += len(common_neighbors)

    # Calculate overall recall
    # Denominator is the total number of true neighbors across all queries
    recall = total_common_neighbors / (num_queries * k)

    return recall
import time
def benchmark_ann():
    N = 4_000_000
    D = pow(2,1)
    K = 5
    Q = 10000
  
    print("Allocating random data...")
    points_cpu = np.random.rand(N, D)
    queries_cpu = np.random.rand(Q, D)
    print("Starting KNN")


    start = time.time()
    indices_knn, _ = knn(queries_cpu, points_cpu, K)
    cp.cuda.Stream.null.synchronize()  # Sync after GPU work is fully done
    end = time.time()
    print(f"KNN took {end - start:.2f} seconds.")
    print("Finished KNN.")

    print("Starting ANN..")
    start = time.time()
    indices_ann = ann(points_cpu, queries_cpu, K)
    cp.cuda.Stream.null.synchronize()  # Sync after GPU work is fully done
    end = time.time()
    print(f"ANN took {end - start:.2f} seconds.")
    print(indices_ann.shape)
    print(indices_knn.shape)
    print("Finished ANN.")
    print(calculate_recall(indices_ann, indices_knn, K))    

benchmark_ann()