import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


class DBSCAN_Analysis:
    """Statistical report of DBSCAN algorithm.

    Every possible value of the hyparameter on the minimum of neighbors
    of a point to be considered as a core point is evaluated to measure
    the performance of the clusters. For this purpose, the optimal value
    of the eps is selected based on the k-distance of kNN algorithm, and
    the Silouette score is calculated.
    """
