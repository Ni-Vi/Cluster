from enum import Enum
from typing import Any, NamedTuple

import numpy as np
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
)
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from tqdm import tqdm, trange


class ClusterName(Enum):
    """gender of annotator."""

    kmeans_inertia = "Inertia"
    kmeans_silhouette = "Silhouette"
    affinity_propagation = "Affinity Propagation"
    gaussian_mixture = "Gaussian Mixture"
    gaussian_mixture_sklearn = "Gaussian Mixture Sklearn"
    optics = "OPTICS"
    hdbscan = "HDBSCAN"
    agglomerative_clustering = "Agglomerative Clustering"


class OptimalClusterLabels(NamedTuple):
    """Number of clusters and the labels for each cluster."""

    cluster_method: ClusterName | Any
    n_clusters: int
    cluster_labels_per_cluster: np.ndarray
    score: float | None


def compute_num_clusters(
    *,
    min_n_cluster: int,
    max_n_cluster: int,
    latent_embeddings: np.ndarray,
) -> list[OptimalClusterLabels]:
    """Comput number of clusters with paired metrics."""
    num_cluster_per_method = []
    gaussian_mixture_sklearn_cluster_numbers = gaussian_mixture_sklearn(
        min_n_cluster=min_n_cluster,
        max_n_cluster=max_n_cluster,
        latent_embeddings=latent_embeddings,
    )

    silhouette_cluster_numbers = kmeans_silhouette_clusters(
        min_n_cluster=min_n_cluster,
        max_n_cluster=max_n_cluster,
        latent_embeddings=latent_embeddings,
    )

    hdbscan_cluster_numbers = hdbscan_clusters(latent_embeddings=latent_embeddings, eps=0.05)
    # eps_list = [0.05, 0.15, 0.25, 0.5, 1.0]
    # for eps in tqdm(eps_list, desc="Calclating OPTICS Clusters"):
    #     optics_clusters_numbers = optics_clusters(latent_embeddings=latent_embeddings, eps=eps)
    #     if optics_clusters_numbers.n_clusters <= 1:
    #         continue
    #     num_cluster_per_method.append(optics_clusters_numbers)
    # optics_clusters_numbers = optics_clusters(latent_embeddings=latent_embeddings, eps=0.05)
    # if optics_clusters_numbers.n_clusters >= 1:
    #     num_cluster_per_method.append(optics_clusters_numbers)

    agglomerative_clusters_numbers = agglomerative_clusters(latent_embeddings=latent_embeddings)
    num_cluster_per_method.extend(gaussian_mixture_sklearn_cluster_numbers)
    num_cluster_per_method.extend(silhouette_cluster_numbers)
    num_cluster_per_method.append(hdbscan_cluster_numbers)
    if (
        agglomerative_clusters_numbers.n_clusters > 1
        and agglomerative_clusters_numbers.n_clusters < len(latent_embeddings)
    ):
        num_cluster_per_method.append(agglomerative_clusters_numbers)
    return num_cluster_per_method


def kmeans_silhouette_clusters(
    *, min_n_cluster: int, max_n_cluster: int, latent_embeddings: np.ndarray
) -> list[OptimalClusterLabels]:
    """Compute the sillhouette score for a range of clusters."""
    silhouette_cluster_metrics = []
    for n_clusters in trange(
        min_n_cluster, max_n_cluster, desc="Training Kmeans through Silhouette"
    ):
        model = KMeans(
            n_clusters=n_clusters, init="k-means++", max_iter=100, n_init=10, random_state=123
        )
        labels = model.fit_predict(latent_embeddings)
        silhouette_score_current = float(silhouette_score(latent_embeddings, labels))
        silhouette_cluster_metrics.append(
            OptimalClusterLabels(
                cluster_method=ClusterName("Silhouette"),
                n_clusters=n_clusters,
                cluster_labels_per_cluster=labels,
                score=round(silhouette_score_current, 3),
            )
        )
    return silhouette_cluster_metrics


def gaussian_mixture_sklearn(
    *,
    min_n_cluster: int,
    max_n_cluster: int,
    latent_embeddings: np.ndarray,
) -> list[OptimalClusterLabels]:
    """Compute clusters through a gaussian mixture modelling approach."""
    n_components = list(range(min_n_cluster, max_n_cluster))
    cluster_metrics = []
    for n in tqdm(n_components, desc="Training GMM - sklearn implementation"):
        model = GaussianMixture(
            n_components=n,
            covariance_type="full",
            init_params="k-means++",
            max_iter=200,
            random_state=123,
        ).fit(latent_embeddings)
        labels = model.predict(latent_embeddings)
        # into float
        bic = round(float(model.bic(latent_embeddings)), 3)
        aic = round(float(model.bic(latent_embeddings)), 3)
        if len(np.unique(labels)) == 1:
            continue
        cluster_metrics.append(
            OptimalClusterLabels(
                cluster_method=ClusterName("Gaussian Mixture Sklearn"),
                n_clusters=n,
                cluster_labels_per_cluster=labels,
                score=None,
            )
        )
    return cluster_metrics


def cluster_inertia(
    min_n_cluster: int,
    max_n_cluster: int,
    latent_embeddings: np.ndarray,
) -> list[OptimalClusterLabels]:
    """Elbow curve for the clustering with optional plotting."""
    cluster_metrics = []

    for i in trange(min_n_cluster, max_n_cluster, desc="Training KMeans through Inertia"):
        kmeans_model = KMeans(
            n_clusters=i, init="k-means++", max_iter=100, n_init=10, random_state=123
        )
        kmeans_model.fit(latent_embeddings)
        inertia_scores = kmeans_model.inertia_
        cluster_labels = kmeans_model.labels_
        cluster_metrics.append(
            OptimalClusterLabels(
                cluster_method=ClusterName("Inertia"),
                n_clusters=i,
                cluster_labels_per_cluster=cluster_labels,
                score=round(inertia_scores, 3),
            )
        )
    return cluster_metrics


def affinity_propagation_clusters(
    latent_embeddings: np.ndarray, preference: int = -100
) -> OptimalClusterLabels:
    """Compute the affinity propagation clusters."""
    # TODO: This needs work
    affinity_propagation = AffinityPropagation(
        random_state=123,
        max_iter=100,
        preference=preference,
    ).fit(latent_embeddings)
    n_clusters = len(affinity_propagation.cluster_centers_indices_)
    labels = affinity_propagation.labels_
    return OptimalClusterLabels(
        cluster_method=ClusterName("Affinity Propagation"),
        n_clusters=n_clusters,
        cluster_labels_per_cluster=labels,
        score=None,
    )


# def optics_clusters(
#     *,
#     latent_embeddings: np.ndarray,
#     min_samples: int = 5,
#     min_cluster_size: float = 0.05,
#     max_eps: float = 0.5,
#     xi: float = 0.05,
#     eps: float = 0.05,
# ) -> OptimalClusterLabels:
#     """Compute the OPTICS clusters."""
#     optics_model = OPTICS(
#         min_samples=min_samples,
#         xi=xi,
#         min_cluster_size=min_cluster_size,
#         max_eps=max_eps,
#     ).fit(latent_embeddings)

#     labels = cluster_optics_dbscan(
#         reachability=optics_model.reachability_,
#         core_distances=optics_model.core_distances_,
#         ordering=optics_model.ordering_,
#         eps=eps,
#     )
#     n_clusters = len(np.unique(labels))

#     # TODO: Watch the tree unfold here!
#     return OptimalClusterLabels(
#         cluster_method=ClusterName("OPTICS"),
#         n_clusters=n_clusters,
#         cluster_labels_per_cluster=labels,
#         score=None,
#     )


def hdbscan_clusters(
    *,
    latent_embeddings: np.ndarray,
    min_samples: int = 5,
    min_cluster_size: int = 5,
    eps: float = 0.25,
) -> OptimalClusterLabels:
    """Compute the OPTICS clusters."""
    labels = HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_epsilon=eps
    ).fit_predict(latent_embeddings)
    n_clusters = len(np.unique(labels))
    return OptimalClusterLabels(
        cluster_method=ClusterName("HDBSCAN"),
        n_clusters=n_clusters,
        cluster_labels_per_cluster=labels,
        score=None,
    )


def agglomerative_clusters(
    latent_embeddings: np.ndarray,
    n_clusters: int = 12,
    distance_threshold: float = 1,
) -> OptimalClusterLabels:
    """Compute the OPTICS clusters."""
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, compute_full_tree=True
    ).fit_predict(latent_embeddings)
    n_clusters = len(np.unique(labels))

    # TODO: Watch the tree unfold here as well1
    return OptimalClusterLabels(
        cluster_method=ClusterName("Agglomerative Clustering"),
        n_clusters=n_clusters,
        cluster_labels_per_cluster=labels,
        score=None,
    )
