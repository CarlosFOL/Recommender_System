import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score, normalized_mutual_info_score, pairwise_distances


class HDBSCAN_Analysis:
    """Statistical report of HDBSCAN algorithm.

    Comparing the performance of the HDBSCAN algorithm by using
    different values of the hyperparameter `min_cluster_size`.
    The main statistics considered in this analysis are:

    * NMI (Normalized Mutual Information): Measures the mutual information
    between clusters and ground truth categories, normalized by their entropies.
    Range [0,1] where higher values indicate better alignment between clustering
    and categories.

    * Persistence scores: Quantifies cluster stability across different
    density levels in the hierarchical structure. Higher values indicate
    more robust clusters that remain stable across multiple density thresholds.

    * Homogeneity: Measures cluster purity by evaluating the extent to which
    each cluster contains elements from a single category. Range [0,1]
    where 1 means perfectly pure clusters.

    * % of noise: Percentage of points classified as noise (-1) that were
    not assigned to any cluster. Typically expected to be 5-20% for
    well-formed clusters in real-world data.

    * Number of clusters: Total number of distinct clusters identified,
    excluding noise.

    Attributes
    ----------
    X: np.ndarray[np.ndarray[float]]
        Inputs for training
    Y: np.ndarray[int]
        Targets

    Methods
    -------
    get_stats(targets: np.ndarray, model:hdbscan.HDBSCAN)
        Calculate the goodness statistics of a particular model

    compare_models(min_sizes: list[int])
        Show a comparison table between HDBSCAN models built with different
        minimum cluster sizes.
    """

    def __init__(self, X:np.ndarray[np.ndarray[float]], Y: np.ndarray[int | str] ):
        self.__X = X
        self.__Y = Y

    @property
    def X(self):
        return self.__X
    @X.setter
    def X(self, new_sample):
        self.X = new_sample

    @property
    def Y(self):
        return self.__Y
    @Y.setter
    def Y(self, new_targets):
        self.Y = new_targets

    @property
    def cat_vars(self):
        return self.__cat_vars


    def get_stats(self, model:hdbscan.HDBSCAN) -> list:
        """
        Calculate the main statistics of a partircular model

        Attributes
        ----------
        model: hdbscan.HDBSCAN
            HDBSCAN model

        Return
        ------
        np.ndarray
            Stats of the model
        """
        pred_class = model.labels_
        nmi = normalized_mutual_info_score(labels_true=self.Y, labels_pred=pred_class)
        pct_noise = np.mean(pred_class == -1) * 100
        n_clusters = len(np.unique(pred_class)) - (pct_noise > 0) # Decrease 1 if label -1 exists.
        homogeneity = homogeneity_score(self.Y, pred_class)
        persistent_score = np.mean(model.cluster_persistence_)
        return [nmi, f"{pct_noise:.2f}%", n_clusters, homogeneity, persistent_score]


    def compare_models(self, min_sizes:list[int]) -> pd.DataFrame:
        """
        Compare the main statistics of the HDBSCAN models built from
        the different values of `min_cluster_size`

        Parameters
        ----------
        min_sizes:list[int]
            'n' min. sizes of clusters to build 'n' DBSCAN Models
            and then compare them.

        Return
        ------
        pd.DataFrame
            Table of statistics
        """
        # Precomputed the distance matrix by using cosine similarity
        df_distances = pairwise_distances(self.X, metric="cosine")
        cols = ["Min_Size", "NMI", "Pct_Noise", "N_Clusters",
                "Homogeneity", "Persistent_Score"]
        df_metrics = []
        for m_size in min_sizes:
            model = hdbscan.HDBSCAN(min_cluster_size=m_size,
                                    metric="precomputed",
                                    core_dist_n_jobs=-1)
            model.fit(df_distances)
            row = [m_size] + self.get_stats(model)
            df_metrics.append(row)
        df_metrics = pd.DataFrame(df_metrics, columns=cols)
        return df_metrics


if __name__ == "__main__":
    X = pd.read_csv("../data/embeddings.csv", nrows=10000)
