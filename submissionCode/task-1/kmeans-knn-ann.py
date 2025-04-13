import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
from sklearn.metrics import rand_score
from sklearn.cluster import KMeans


####################
# GPU BENCHMARKING #
####################

def gpu_stats():
    """
    Get GPU utilization using nvidia-smi command.

    :returns: GPU utilization as a string.
    """
    query_metrics = ["utilization.gpu"]
    metrics_str = ",".join(query_metrics)
    command = [
        "nvidia-smi",
        f"--query-gpu={metrics_str}",
        "--format=csv,noheader,nounits",
        f"--id=0"
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


gpu_utils = []


#############################
# K-MEANS UTILITY FUNCTIONS #
#############################

def refine_clusters(p_gpu, current_label_index):
    """
    Compute new centroids by averaging the points in each cluster.

    :param p_gpu: Points on GPU.
    :param current_label_index: Current cluster labels.
    :returns: New centroids.
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


def initialize_centers(data, k):
    """
    Randomly select k points from the data to initialize the cluster centers.

    :param data: Input data.
    :param k: Number of clusters.
    :returns: Randomly selected points as initial cluster centers.
    """
    idx = np.random.choice(len(data), size=k, replace=False)
    return cp.asarray(data[idx])


def assign_labels(data, centers, batch_size=100_000):
    """
    Assign each point in the data to the nearest cluster center.
    This is done in batches to avoid memory issues.

    :param data: Input data.
    :param centers: Cluster centers.
    :param batch_size: Size of each batch for processing.
    :returns: Array of labels indicating the cluster each point belongs to.
    """
    # preallocate labels
    labels = np.empty(data.shape[0])
    for start_idx in tqdm(range(0, data.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, data.shape[0])
        batch = cp.asarray(data[start_idx:end_idx])
        dists = cp.linalg.norm(batch[:, None, :] - centers[None, :, :], axis=2)
        labels[start_idx:end_idx] = cp.argmin(dists, axis=1).get()
    return labels


###########
# K-MEANS #
###########

def k_means_mini_batch(data, k, iterations=10, batch_size=2048):
    """
    Perform K-Means clustering using mini-batches.
    This function initializes k cluster centers, and iteratively refines them
    by assigning points to the nearest center and updating the centers.
    The process is repeated for a specified number of iterations.
    The function uses a batch size to limit the number of points processed at once.

    :param data: Input data.
    :param k: Number of clusters.   
    :param iterations: Number of iterations to perform.
    :param batch_size: Size of each batch for processing.
    :returns: Final cluster centers and labels for each point.
    """
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
    """
    Perform K-Means clustering on the given data using a GPU.
    This function initializes c cluster centers and iteratively refines them
    by assigning points to the nearest center and updating the centers.
    The process is repeated for a specified number of iterations.
    The function uses a batch size to limit the number of points processed at once.

    :param p_cpu: Input data on CPU.
    :param c: Number of clusters.
    :param distance_fn: Distance function to use.
    :param max_iter: Number of iterations to perform.
    :returns: Final cluster centers and labels for each point.
    """
    n_p = p_cpu.shape[0]
    centroids = p_cpu[np.random.randint(0, n_p, (c,))]
    indices, _ = knn(p_cpu, centroids, 1)
    indices = indices.flatten()
    p_gpu = cp.asarray(p_cpu)
    for i in tqdm(range(max_iter)):
        centroids = refine_clusters(p_gpu, indices)
        new_indices, new_distances = knn(
            p_cpu, cp.asnumpy(centroids), 1, distance_fn)
        new_indices = new_indices.flatten()
        gpu_utils.append(gpu_stats())
        indices = new_indices
    return centroids, indices


#############################
# KNN/ANN UTILITY FUNCTIONS #
#############################

def euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two matrices.
    Each row of the first matrix is compared to each row of the second matrix.

    :param a: First matrix.
    :param b: Second matrix.
    :returns: Matrix of distances.
    """
    return cp.linalg.norm(a - b, axis=-1)


def group_by_cluster(points, labels):
    """
    Groups points by their cluster labels.
    This function sorts the points and labels, and then groups them based on the labels.
    It returns a list of tuples, where each tuple contains the points and their corresponding labels.
    The points and labels are sorted in ascending order.

    :param points: Points to be grouped.
    :param labels: Labels corresponding to the points.
    :returns: List of tuples containing grouped points and labels.
    """
    sorted_label_indices = cp.argsort(labels)
    sorted_labels = labels[sorted_label_indices]
    sorted_points = points[sorted_label_indices]
    change_indices = cp.where(cp.diff(sorted_labels) != 0)[0] + 1
    start_indices = cp.concatenate((cp.array([0]), change_indices)).get()
    end_indices = cp.concatenate(
        (change_indices, cp.array([len(labels)]))).get()
    return [(sorted_points[start:end], sorted_label_indices[start:end]) for start, end in zip(start_indices, end_indices)]


###########
# KNN/ANN #
###########

def knn(q_cpu, p_cpu, k, distance_fn=euclidean_distance, batch_size=1_000_000):
    """
    Find the k nearest neighbors for each point in q_cpu from p_cpu.
    This is done using a GPU for the query points and CPU for the points.
    The function uses a batch size to limit the number of points processed at once.

    :param q_cpu: Query points on CPU.
    :param p_cpu: Points on CPU.
    :param k: Number of nearest neighbors to find.
    :param distance_fn: Distance function to use.
    :param batch_size: Size of each batch for processing.
    :returns: Indices and distances of the k nearest neighbors.
    """
    def _knn(q_gpu, p_cpu, k, distance_fn, batch_size=8192):
        """
        Find the k nearest neighbors for each point in q_gpu from p_cpu.
        This is done using a GPU for the query points and CPU for the points.
        The function uses a batch size to limit the number of points processed at once.

        :param q_gpu: Query points on GPU.
        :param p_cpu: Points on CPU.
        :param k: Number of nearest neighbors to find.
        :param distance_fn: Distance function to use.
        :param batch_size: Size of each batch for processing.
        :returns: Indices and distances of the k nearest neighbors.
        """
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
            batch_indices_expanded = cp.tile(
                batch_indices_abs[None, :], (n_q, 1))

            concatenated_distances = cp.concatenate(
                (overall_top_k_distance, batch_distances), axis=1)
            concatenated_indices = cp.concatenate(
                (overall_top_k_indices, batch_indices_expanded), axis=1)

            if not (start_idx % 100):
                gpu_utils.append(gpu_stats())

            top_k_indices = cp.argpartition(
                concatenated_distances, k, axis=1)[:, :k]
            overall_top_k_distance = cp.take_along_axis(
                concatenated_distances, top_k_indices, axis=1)
            overall_top_k_indices = cp.take_along_axis(
                concatenated_indices, top_k_indices, axis=1)

        return overall_top_k_distance, overall_top_k_indices

    # Determine the ratio of batch size for queries and points
    n_q = q_cpu.shape[0]
    q_batch_size = n_q

    indices = np.empty((n_q, k), dtype=np.int32)
    distances = np.empty((n_q, k), dtype=np.float32)

    for start_idx in tqdm(range(0, n_q, q_batch_size)):
        end_idx = min(start_idx + batch_size, n_q)
        q_gpu = cp.asarray(q_cpu[start_idx:end_idx])
        batch_distances, batch_indices = _knn(
            q_gpu, p_cpu, k, distance_fn, 1_000_000)
        indices[start_idx:end_idx] = cp.asnumpy(batch_indices)
        distances[start_idx:end_idx] = cp.asnumpy(batch_distances)
    return indices, distances


def ann(p_cpu, q_cpu, k, c, distance_fn):
    """
    Approximate Nearest Neighbor search using K-Means clustering.
    This function first clusters the points in p_cpu into c clusters using K-Means.
    Then, it finds the nearest centroid for each query point in q_cpu.
    Finally, it finds the nearest points in each cluster for the query points.
    The results are returned in the original order of the query points.

    :param p_cpu: Points on CPU.
    :param q_cpu: Query points on CPU.
    :param k: Number of nearest neighbors to find.
    :param c: Number of clusters.
    :param distance_fn: Distance function to use.
    :returns: Indices of the k nearest neighbors for each query point.
    """
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


################
# BENCHMARKING #
################

def calculate_recall(ann_indices, knn_indices, k):
    """
    Calculate the recall of the ANN indices compared to the KNN indices.
    Recall is defined as the number of true neighbors found by ANN divided by the total number of true neighbors.

    :param ann_indices: Indices of the ANN results.
    :param knn_indices: Indices of the KNN results.
    :param k: Number of neighbors.
    :returns: Recall score.
    """
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
        return 1.0  # Perfect recall if there are no queries to test
    if k == 0:
        return 0.0  # Cannot recall anything if k=0

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


def compare_kmeans(reference, test):
    """
    Compare two KMeans results using RandIndex.
    The RandIndex is a measure of the similarity between two data clusterings.
    It is defined as the number of pairs of points that are either in the same cluster or in different clusters in both clusterings.

    :param reference: Reference clustering labels.
    :param test: Test clustering labels.
    :returns: RandIndex score.
    """
    # Compare the two KMeans results using RandIndex
    reference_labels = cp.asnumpy(reference)
    test_labels = cp.asnumpy(test)
    return rand_score(reference_labels, test_labels)


def benchmark_ann(N, Ds, K, Q, distance_fn):
    """
    Benchmark the ANN algorithm with different cluster sizes and dimensions.
    This function generates random points and queries, and measures the time taken
    to find the nearest neighbors using both KNN and ANN.
    It also calculates the recall score for the ANN results compared to the KNN results.
    The results are plotted for visualization.

    :param N: Number of points.
    :param Ds: List of dimensions.
    :param K: Number of nearest neighbors to find.
    :param Q: Number of queries.
    :param distance_fn: Distance function to use.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.subplots_adjust(wspace=1)
    ax1 = axes[0]
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Time Taken (seconds)")
    ax1.set_title("ANN: Execution Time vs. Recall Score")
    ax2 = axes[1]
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("Recall Score")
    ax2.set_title("Mini-Batch K-Means: Accuracy (Rand Index) vs. Batch Size")
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

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

        ax1.plot(
            cluster_sizes,
            time_taken,
            marker="o",
            linestyle="-",
            label=f"D={D}")
        ax2.plot(
            cluster_sizes,
            recall_score,
            marker="s",
            linestyle="-",
            label=f"D={D}")

    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.savefig(f"benchmark_ann_{N}_{"-".join(map(str, Ds))}.png")


def benchmark_kmeans(N, Ds, C):
    """
    Benchmark the KMeans algorithm with different dimensions and batch sizes.
    This function generates random points and measures the time taken
    to find the nearest neighbors using both KMeans and Mini-Batch KMeans.
    The results are plotted for visualization.

    :param N: Number of points.
    :param Ds: List of dimensions.
    :param C: Number of clusters.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.subplots_adjust(wspace=1)
    ax1 = axes[0]
    ax1.set_xscale("log")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time Taken (seconds)")
    ax1.set_title("Mini-Batch K-Means: Execution Time vs. Batch Size")
    ax2 = axes[1]
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Rand Index (vs Reference)")
    ax2.set_title("Mini-Batch K-Means: Accuracy (Rand Index) vs. Batch Size")
    # Adjust layout to prevent overlapping titles/labels
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

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
            centroids, labels = k_means_mini_batch(
                points_cpu,
                C,
                iterations=10,
                batch_size=b
            )
            # Sync after GPU work is fully done
            cp.cuda.Stream.null.synchronize()
            end = time.time()
            time_taken.append(end - start)
            rand_index.append(compare_kmeans(reference_labels, labels))

        ax1.plot(
            batch_size,
            time_taken,
            marker="o",
            linestyle="-",
            label=f"D={D}"
        )
        ax2.plot(
            batch_size,
            rand_index,
            marker="s",
            linestyle="-",
            label=f"D={D}"
        )

    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.savefig(f"benchmark_kmeans_{N}_{"-".join(map(str, Ds))}_{C}.png")


benchmark_kmeans(
    4_000_000,
    [2 ** i for i in range(4)],
    32
)
benchmark_ann(
    [4_000_000],
    [2 ** i for i in range(7)],
    32,
    10,
    euclidean_distance
)
