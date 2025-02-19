import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    sample: np.ndarray[np.ndarray[float]]
        Training set
    k_min: int
        Lowest number of neighbors so that a point to be considered a
        core point.
    k_max: int
        Highest number of neighbors so that a point to be considered a
        core point
    """

    def __init__(self, sample: np.ndarray[np.ndarray[float]],
                 k_min: int, k_max: int = None):
        self.__X = sample
        self.__k_min = k_min
        if k_max is None:
            self.__k_max = k_min + 5
        elif (k_max - k_min) + 1 > 9:
            # To avoid plotting too many graphs.
            self.__k_max = k_min + 9

    @property
    def X(self):
        return self.__X
    @X.setter
    def X(self, new_sample):
        self.X = new_sample

    @property
    def k_min(self) -> int:
        return self.__k_min
    @k_min.setter
    def k_min(self, new_kmin):
        if new_kmin >= self.k_max:
            self.k_max = new_kmin + 9
        self.__k_min = new_kmin

    @property
    def k_max(self) -> int:
        return self.__k_max
    @k_max.setter
    def k_max(self, new_kmax):
        if new_kmax <= self.k_min:
            self.k_min = max(2, new_kmax - 10)
        self.__k_max = new_kmax

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
        knn = NearestNeighbors(n_neighbors=k, metric="cosine")
        knn.fit(self.X)
        distances, _ = knn.kneighbors(self.X)
        # Remove the distance between a point and itself.
        distances = np.sort(distances.reshape(-1,))
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
        fig, axes = plt.subplots(2, columns, figsize = (18, 4))
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

    def hyperparameters_search(self, min_samples: np.ndarray, eps: np.ndarray):
        """
        Calculate the silhouette score for each combination of pairs of
        eps and k, to get the best hyperparameters for DBSCAN.

        Parameters
        ----------
        min_samples: np.ndarray
            Minimum number of neighbors to be considered a core point.
        eps: np.ndarray
            Maximum distance between two points to be neighbors.
        """
        summary_table = pd.DataFrame(index=min_samples, columns=eps.round(3), dtype=float)
        for i in range(len(min_samples)):
            row = []
            for e in eps:
                model = DBSCAN(min_samples=min_samples[i], eps=e, metric="cosine")
                labels = model.fit(self.X).labels_
                row.append(silhouette_score(self.X, labels))
            summary_table.iloc[i, :] = row
        # Plot
        fig = plt.figure(figsize=(11, 5))
        axes = fig.add_axes([0, 0, 1, 1])
        sns.heatmap(summary_table, annot = True, ax=axes)
        axes.set_title("Silhouette score for each pair of hyperparameters")
        plt.show()


if __name__ == "__main__":
    X = pd.read_csv("../data/embeddings.csv", nrows=10000)
    model = DBSCAN_Analysis(sample=X, k_min=4)
    min_samples = np.arange(4, 10); eps = np.linspace(0.1, 0.15, 10)
    sum_table = model.hyperparameters_search(min_samples, eps)
