from test import testdata_kmeans, testdata_knn, testdata_ann
from distances import distance_cosine_CUPY, distance_l2_CUPY, distance_dot_CUPY, distance_manhattan_CUPY
from ann import knn, k_means, ann


# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    return distance_cosine_CUPY(X, Y)


def distance_l2(X, Y):
    return distance_l2_CUPY(X, Y)


def distance_dot(X, Y):
    return distance_dot_CUPY(X, Y)


def distance_manhattan(X, Y):
    return distance_manhattan_CUPY(X, Y)


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K):
    return knn(X[None, ...], A, K)


# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K):
    return k_means(A, K)


# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K):
    return ann(X[None, ...], A, K, 4)


# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
# def test_kmeans():
#     N, D, A, K = testdata_kmeans("test_file.json")
#     kmeans_result = our_kmeans(N, D, A, K)
#     print(kmeans_result)

# def test_knn():
#     N, D, A, X, K = testdata_knn("test_file.json")
#     knn_result = our_knn(N, D, A, X, K)
#     print(knn_result)

# def test_ann():
#     N, D, A, X, K = testdata_ann("test_file.json")
#     ann_result = our_ann(N, D, A, X, K)
#     print(ann_result)

# def recall_rate(list1, list2):
#     """
#     Calculate the recall rate of two lists
#     list1[K]: The top K nearest vectors ID
#     list2[K]: The top K nearest vectors ID
#     """
#     return len(set(list1) & set(list2)) / len(list1)

# if __name__ == "__main__":
#     test_kmeans()
