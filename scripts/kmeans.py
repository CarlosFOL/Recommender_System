from math import dist, sqrt
import pandas as pd
from sklearn.cluster import KMeans


def _BSS(X):
    """
    Calculate the Between Cluster Sum of Squares.

    Parameters
    ----------
    X: pd.DataFrame
        Training data

    model: sklearn.cluster.KMeans
        K-means model
    """
    bar_X = X.mean()
    def get(model, k: int) -> float:
        """
        Return the BSS according to particular k

        Parameters
        ----------
        k: int
            Number of clusters
        """
        value = 0
        labels = model.labels_
        for i in range(k):
            C_i = X.iloc[labels == i, :]
            centroid_i = C_i.mean()
            value += C_i.shape[0] * dist(centroid_i, bar_X)**2
        return value
    return get


def kmeans_statistics(X: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Compute the BSS and WSS for different values of k
    in order to get the best one for this algorithm.

    Parameters
    ----------
    X: pd.DataFrame
        A sample or total training data to use in k-means
    """
    n = X.shape[0] 
    get_BSS = _BSS(X)
    # Stats
    WSS = []
    BSS = []
    upper_limit = round(sqrt(n)) # Square root as heuristic rule
    for k in range(2, upper_limit): 
        model = KMeans(n_clusters = k, random_state=123)
        model.fit(X)
        WSS.append(model.inertia_)
        BSS.append(get_BSS(model, k))
    return WSS, BSS

if __name__ == "__main__":
    df = pd.read_csv("../data/users.csv", nrows=10000)
    WSS, BSS = kmeans_statistics(df)
    kmeans_summary = pd.DataFrame({"WSS": WSS, "BSS": BSS})
    print(kmeans_summary)