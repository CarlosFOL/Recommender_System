from math import dist, sqrt
import matplotlib.pyplot as plt
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

def kmeans_statistics(X: pd.DataFrame,
                      upper_limit:float = None) -> tuple[list[float], list[float]]:
    """
    Compute the BSS and WSS for different values of k in order to get
    the best one for this algorithm.

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
    if upper_limit is None or type(upper_limit) != int:
        upper_limit = round(sqrt(n)) # Square root as heuristic rule
    for k in range(2, upper_limit + 1):
        model = KMeans(n_clusters = k, random_state=123)
        model.fit(X)
        WSS.append(model.inertia_)
        BSS.append(get_BSS(model, k))
    return WSS, BSS

def plot_kmeans_stats(df_kmeans: pd.DataFrame):
    """
    Apply the elbow method to find out the best k according to the WSS
    and BSS

    Parameters
    ----------
    df_kmeans: pd.DataFrame
        Contain the values of the statisticians.
    """
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize = (10, 3))
    for ax, stat in [(ax1, "WSS"), (ax2, "BSS")]:
        ax.plot(df_kmeans["K"], df_kmeans[stat])
        ax.set_title(f"Elbow: {stat}")
        ax.set_xlabel("K")
        ax.set_ylabel(stat)
    plt.show()

def centroid_clusters(model: KMeans, features: list):
    """
    Show the features of the clusters
    centroids

    Parameters
    ----------
    model: KMeans
        Trained k-means model
    """
    k = model.n_clusters
    centroids = pd.DataFrame(model.cluster_centers_, columns = features)
    centroids["Cluster size"] = [sum(model.labels_ == i) for i in range(k)]
    return centroids


if __name__ == "__main__":
    df = pd.read_csv("../data/users.csv", nrows=10000)
    WSS, BSS = kmeans_statistics(df)
    kmeans_summary = pd.DataFrame({"WSS": WSS, "BSS": BSS})
    print(kmeans_summary)
