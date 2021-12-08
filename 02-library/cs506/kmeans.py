from collections import defaultdict
from math import inf
import random
import csv

import numpy as np

# def point_avg(points):
#     """
#     Accepts a list of points, each with the same number of dimensions.
#     (points can have more dimensions than 2)

#     Returns a new point which is the center of all the points.
#     """
#     raise NotImplementedError()


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    unique_k_list = list(set(assignments))
    centers = []
    for k in unique_k_list:
        index_list = [idx for idx, i in enumerate(assignments) if i == k]
        centers.append(np.array(dataset)[index_list, :].mean(axis=0).tolist())
    return centers


def assign_points(data_points, centers):
    """ """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    res = 0
    for i in range(len(a)):
        res += (np.array(a[i]) - np.array(b)) ** 2
    res = np.sum(res)
    return res ** (1 / 2)


# def distance_squared(a, b):
#     raise NotImplementedError()


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    idxs = random.choices(list(range(0, len(dataset))), k=k)
    res = [dataset[idx] for idx in idxs]
    # res = [random.randrange(0, k, 1) for i in range(len(dataset))]
    return res


def cost_function(clustering):
    import numpy as np

    costs = []
    for k, v_list in clustering.items():
        costs.append(np.sum(np.sqrt(np.square(np.array(v_list) - np.mean(v_list, axis=0)))))
    return np.sum(costs)


# def generate_k_pp(dataset, k):
#     """
#     Given `data_set`, which is an array of arrays,
#     return a random set of k points from the data_set
#     where points are picked with a probability proportional
#     to their distance as per kmeans pp
#     """
#     raise NotImplementedError()


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")
    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
