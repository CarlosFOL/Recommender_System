import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
   
    * DBCV (Density-Based Clustering Validation): Evaluates density-based
    clustering quality by measuring both internal cluster density and
    separation between clusters. Range [-1,1] where higher values 
    indicate better clustering.
    
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
    sample: np.ndarray[np.ndarray[float]]
        Training set
    cat_vars: pd.DataFrame
        Categorical variables in dataset
    """

    def __init__(self, sample:np.ndarray[np.ndarray[float]], cat_vars:pd.DataFrame):
        self.__X = sample
        self.__cat_vars = cat_vars
    
    @property
    def X(self):
        return self.__X
    @X.setter
    def X(self, new_sample):
        self.X = new_sample

    @property
    def cat_vars(self):
        return self.__cat_vars

    def get_stats(self, true_values:np.ndarray, model:hdbscan.HDBSCAN) -> list:
        """
        Calculate the main statistics of a partircular model
        Attributes
        ----------
        true_values: np.ndarray
            The true classes of each data point.
        model: hdbscan.HDBSCAN
            HDBSCAN model
        
        Return
        ------
        np.ndarray
            Stats of the model 
        """
        pred_values = model.labels_
        nmi = normalized_mutual_info_score(labels_true=true_values, labels_pred=pred_values)
        pct_noise = np.mean(pred_values == -1) * 100
        n_clusters = np.unique(pred_values) - (pct_noise > 0) # Decrease 1 if label -1 exists.
        dbcv = model.relative_validity_
        homogeneity = homogeneity_score(true_values, pred_values)
        persistent_score = np.mean(model.cluster_persistence_)
        return [nmi, pct_noise, n_clusters, dbcv, homogeneity, persistent_score]

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
                "DBCV", "Homogeneity", "Persistent_Score"]
        tables = []
        for cat in self.cat_vars:
            df_stats = []
            for m_size in min_sizes:
                model = hdbscan.HDBSCAN(min_cluster_size=m_size, 
                                        metric="precomputed",
                                        core_dist_n_jobs=-1)
                model.fit(df_distances)
                row = [m_size] + self.get_stats(cat, model)
                df_stats.append(row)
            tables.append(pd.DataFrame(df_stats, columns=cols))
        return tables


if __name__ == "__main__":
    X = pd.read_csv("../data/embeddings.csv", nrows=10000)