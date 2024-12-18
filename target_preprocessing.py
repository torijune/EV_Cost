import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def categorize_by_quantiles(df, column, labels):
    categorized = pd.qcut(df[column], q=len(labels), labels=labels, retbins=True)
    # print(f"Quantile boundaries: {categorized[1]}")
    return categorized[0]

def categorize_by_mean_std(df, column):
    mean = df[column].mean()
    std = df[column].std()
    # print(f"Mean: {mean}, Std: {std}")
    # print(f"Boundaries: Low <= {mean - std}, Mid = ({mean - std}, {mean + std}], High > {mean + std}")
    
    def categorize(value):
        if value <= mean - std:
            return '저가'
        elif mean - std < value <= mean + std:
            return '중가'
        else:
            return '고가'

    return df[column].apply(categorize)

def categorize_by_kmeans(df, column, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(df[[column]])
    centroids = model.cluster_centers_.flatten()
    print(f"K-Means centroids: {centroids}")

    cluster_means = {i: centroids[i] for i in range(n_clusters)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    cluster_map = {cluster: label for cluster, label in zip(sorted_clusters, ['저가', '중가', '고가'])}
    
    # print(f"Cluster mapping: {cluster_map}")
    return pd.Series(clusters).map(cluster_map)

def categorize_by_gmm(df, column, n_components=3):
    model = GaussianMixture(n_components=n_components, random_state=42)
    clusters = model.fit_predict(df[[column]])
    means = model.means_.flatten()
    print(f"GMM means: {means}")

    cluster_means = {i: means[i] for i in range(n_components)}
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    cluster_map = {cluster: label for cluster, label in zip(sorted_clusters, ['저가', '중가', '고가'])}
    
    # print(f"Cluster mapping: {cluster_map}")
    return pd.Series(clusters).map(cluster_map)


