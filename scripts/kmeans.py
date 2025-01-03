from sklearn.cluster import KMeans
from math import dist, sqrt

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
            BSS += C_i.shape[0] * dist(centroid_i, bar_X)**2
        return value
    return get


def kmeans_statistics(X):
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
    for k in range(2, sqrt(n)): # Square root as heuristic rule
        model = KMeans(n_clusters = k, random_state=123)
        model.fit(X)
        WSS.append(model.inertia_)
        BSS.append(get_BSS(model, k))
    return WSS, BSS