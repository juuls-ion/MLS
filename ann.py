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

def overall_knn(q_gpu, p_cpu, k, distance_fn, batch_size=8192):
    n_q = q_gpu.shape[0]
    n_p = p_cpu.shape[0]
    global gpu_utils

    # Preallocate arrays
    overall_top_k_distance = cp.full((n_q, k), cp.inf)
    overall_top_k_indices = cp.full((n_q, k), -1)

    for start_idx in range(0, n_p, batch_size):

        end_idx = min(start_idx + batch_size, n_p)

        p_gpu = cp.asarray(p_cpu[start_idx:end_idx])
        batch_distances = distance_fn(q_gpu[:, None, :], p_gpu[None, :, :])

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


def euclidean_distance(a, b):
    return cp.linalg.norm(a - b, axis=-1)


#@cp.fuse('knn')
def knn(q_cpu, p_cpu, k, distance_fn=euclidean_distance, batch_size=10_000_00):
    # Determine the ratio of batch size for queries and points
    n_q = q_cpu.shape[0]
    n_p = p_cpu.shape[0]
    
    p_batch_size = int(batch_size / n_q) 
 
    q_batch_size = n_q

    indices = np.empty((n_q, k), dtype=np.int32)
    distances = np.empty((n_q, k), dtype=np.float32)

    for start_idx in tqdm(range(0, n_q, q_batch_size)):
        end_idx = min(start_idx + batch_size, n_q)
        q_gpu = cp.asarray(q_cpu[start_idx:end_idx])
        batch_distances, batch_indices = overall_knn(q_gpu, p_cpu, k, distance_fn, p_batch_size)
        indices[start_idx:end_idx] = cp.asnumpy(batch_indices)
        distances[start_idx:end_idx] = cp.asnumpy(batch_distances)
    return indices, distances

def initialize_centers(data, k):
    idx = np.random.choice(len(data), size=k, replace=False)
    return cp.asarray(data[idx])

def assign_labels(data, centers, batch_size=1_000_0):
    # preallocate labels
    labels = np.empty(data.shape[0])   
    for start_idx in tqdm(range(0, data.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, data.shape[0])
        batch = cp.asarray(data[start_idx:end_idx])
        dists = cp.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
        labels[start_idx:end_idx] = cp.argmin(dists, axis=1).get()
    return labels

def k_means_mini_batch(data, k, iterations=10, batch_size=2048):
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
    labels = assign_labels(data, centers)
    return centers, labels

def k_means(p_cpu, c, distance_fn, max_iter=10):
    n_p = p_cpu.shape[0]
    centroids = p_cpu[np.random.randint(0, n_p, (c,))]
    indices, _ = knn(p_cpu, centroids, 1)
    indices = indices.flatten()
    p_gpu = cp.asarray(p_cpu)
    for i in tqdm(range(max_iter)):
        centroids = refine_clusters(p_gpu, indices)
        new_indices, new_distances = knn(p_cpu, cp.asnumpy(centroids), 1, distance_fn)
        new_indices = new_indices.flatten()
        gpu_utils.append(gpu_stats())
        indices = new_indices
    return centroids, indices

def group_by_cluster(points, labels):
    sorted_label_indices = cp.argsort(labels)
    sorted_labels = labels[sorted_label_indices]
    sorted_points = points[sorted_label_indices]
    change_indices = cp.where(cp.diff(sorted_labels) != 0)[0] + 1
    start_indices = cp.concatenate((cp.array([0]), change_indices)).get()
    end_indices = cp.concatenate((change_indices, cp.array([len(labels)]))).get()
    return [(sorted_points[start:end], sorted_label_indices[start:end]) for start, end in zip(start_indices, end_indices)]

def ann(p_cpu, q_cpu, k, c, distance_fn):
    centroids, labels = k_means_mini_batch(p_cpu, c, iterations=10)
    grouped_p = group_by_cluster(p_cpu, labels)
    # Now, find the nearest centroid for each query
    q_gpu = cp.asarray(q_cpu)
    indices, _ = knn(q_gpu, centroids, 1, distance_fn)
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

def compare_kmeans(reference, test):
    # Compare the two KMeans results using RandIndex
    from sklearn.metrics import rand_score
    reference_labels = cp.asnumpy(reference)
    test_labels = cp.asnumpy(test)
    return rand_score(reference_labels, test_labels)   

def benchmark_ann(N, Ds, K, Q, distance_fn):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.subplots_adjust(wspace=1)
    ax1 = axes[0]
    ax1.set_ylabel('Time Taken (seconds)')
    ax1.set_title('ANN: Execution Time vs. Recall Score')
    ax2 = axes[1]
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Recall Score')
    # ax2.set_ylim(0, 1.05) # Optional: Set limits if Rand Index is always 0-1
    ax2.set_title('Mini-Batch K-Means: Accuracy (Rand Index) vs. Batch Size')
    ax2.set_xscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Cluster Size')
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle

    for D in Ds:
        points_cpu = np.random.rand(N, D)
        queries_cpu = np.random.rand(Q, D)
   
        start = time.time()
        indices_knn, _ = knn(queries_cpu, points_cpu, K)
        cp.cuda.Stream.null.synchronize()  # Sync after GPU work is fully done
        end = time.time()

        print(f"KNN took {end - start:.2f} seconds.")
  
        cluster_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        recall_score = []
        time_taken = []
        for c in cluster_sizes:
            start = time.time()
            indices_ann = ann(points_cpu, queries_cpu, K, c, distance_fn)
            cp.cuda.Stream.null.synchronize()  # Sync after GPU work is fully done
            end = time.time()
            recall_score.append(calculate_recall(indices_ann, indices_knn, K)) 
            time_taken.append(end - start)

        ax1.plot(cluster_sizes, time_taken, marker='o', linestyle='-', label=f'D={D}')
        ax2.plot(cluster_sizes, recall_score, marker='s', linestyle='-', label=f'D={D}')
    
    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.savefig(f"benchmark_ann_{N}_{"-".join(map(str, Ds))}.png")

def benchmark_kmeans(N, Ds, C):
    from sklearn.cluster import KMeans

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.subplots_adjust(wspace=1)
    ax1 = axes[0]
    ax1.set_ylabel('Time Taken (seconds)')
    ax1.set_title('Mini-Batch K-Means: Execution Time vs. Batch Size')
    ax2 = axes[1]
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Rand Index (vs Reference)')
    # ax2.set_ylim(0, 1.05) # Optional: Set limits if Rand Index is always 0-1
    ax2.set_title('Mini-Batch K-Means: Accuracy (Rand Index) vs. Batch Size')
    ax2.set_xscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Batch Size')
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle

    for D in Ds:
        # Compare KMeans with KMeansMiniBatch speed and randindex score
        points_cpu = np.random.rand(N, D)
        # reference_centroids, reference_labels = k_means(points_cpu, C, max_iter=100)
        reference_kmeans = KMeans(n_clusters=C, n_init=10, max_iter=100)
        reference_kmeans.fit(points_cpu)
        reference_centroids = reference_kmeans.cluster_centers_
        reference_labels = reference_kmeans.labels_
    
        batch_size = [10, 100, 1000, 10_000, 100_000, 1_000_000]
        time_taken = []
        rand_index = []
        for b in tqdm(batch_size):
            start = time.time()
            centroids, labels = k_means_mini_batch(points_cpu, C, iterations=10, batch_size=b)
            cp.cuda.Stream.null.synchronize()  # Sync after GPU work is fully done
            end = time.time()
            time_taken.append(end - start)
            rand_index.append(compare_kmeans(reference_labels, labels))

        ax1.plot(batch_size, time_taken, marker='o', linestyle='-', label=f'D={D}')
        ax2.plot(batch_size, rand_index, marker='s', linestyle='-', label=f'D={D}')
    
    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.savefig(f"benchmark_kmeans_{N}_{"-".join(map(str, Ds))}_{C}.png")

benchmark_ann(4_000_000, [2 ** i for i in range(7)], 32, 1_0, euclidean_distance)
benchmark_kmeans(4_000_000, [2 ** i for i in range(4)], 32)
