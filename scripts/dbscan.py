import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class DBSCAN_Analysis:
    """Statistical report of DBSCAN algorithm.

    Every possible value of the hyparameter on the minimum of neighbors
    of a point to be considered as a core point is evaluated to measure
    the performance of the clusters. For this purpose, the optimal value
    of the eps is selected based on the k-distance of kNN algorithm, and
    the Silouette score is calculated.

    Attributes
    ----------
    k_min: int
        Lowest number of neighbors so that a point to be considered a
        core point.
    k_max: int
        Highest number of neighbors so that a point to be considered a
        core point
    """

    def __init__(self, sample: np.ndarray[np.ndarray[float]],
                 k_min: int, k_max: int = None):
        self.X = sample
        self.k_min = k_min
        if k_max is None:
            self.k_max = k_min + 5
        elif (k_max - k_min) + 1 > 10:
            # To avoid plotting too many graphs.
            self.k_max = 10

    def _kdistance(self, k: int) -> np.ndarray[float]:
        """
        Calculate the distances between each point and its corresponding
        neighbors.

        Parameters
        ----------
        k: int
            Number of neighbors in the kNN algorithm.

        Return
        ------
        np.array[float]
            1D Array of the distances.
        """
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(self.X)
        distances, _ = knn.kneighbors(self.X)
        # Remove the distance between a point and itself.
        distances = np.sort(distances[distances > 1e-3])
        return distances


    def suitable_eps_search(self, elbow: float = 0):
        """
        Get the optimal value for eps by plotting the k-distances and
        recognizing the 'elbow'.

        Parameters
        ----------
        elbow: float
            The elbow to set in all the graphs.
        """
        graphs = (self.k_max - self.k_min) + 1
        columns = round(graphs / 2)
        r, c = (0, 0)
        step = 0
        fig, axes = plt.subplots(2, columns, figsize = (15, 4))
        fig.suptitle("K-distance")
        for k in range(self.k_min, self.k_max + 1):
            distances = self._kdistance(k)
            X = np.arange(distances.shape[0])
            axes[r][c + step].plot(X, distances)
            axes[r][c + step].plot(X, np.ones(X.shape[0])*elbow, "r--")
            axes[r][c + step].set(xlabel = "Points", ylabel=(f"{k}NN-distance"))
            if step == columns - 1:
                r += 1
                c = 0
                step = 0
            else:
                step += 1
        plt.show()
