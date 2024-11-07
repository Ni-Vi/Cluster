from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from mtl_cluster.cluster_related.metrics.compute_n_clusters import (
    ClusterName,
    OptimalClusterLabels,
)


class ClusteringPerformance(BaseModel):
    """Clustering internal performance metrics."""

    clustering_method: ClusterName
    n_clusters: int
    purity_political: float
    # purity_socioecon: float
    purity_edulevel: float
    silhouette_score: float
    predicted_labels: np.ndarray
    num_cluster_score: float | None
    davies_bouldin_score: float
    calinksi_harabasz_score: float
    inter_method_metrics_political: dict[int, dict[str, Any]]
    inter_method_metrics_edulevel: dict[int, dict[str, Any]]

    class Config:
        """Config."""

        arbitrary_types_allowed = True


# Internal Validation Metrics
def cluster_numbers_validation_scores(
    *,
    cluster: OptimalClusterLabels,
    latent_embeddings: np.ndarray,
    true_labels_party: np.ndarray,
    # true_labels_socioecon:np.array,
    true_labels_edulevel: np.ndarray,
) -> ClusteringPerformance:
    """Compute number of clusters and paired internal validation metrics."""
    silhouette = round(
        float(
            silhouette_score(
                latent_embeddings, cluster.cluster_labels_per_cluster, metric="euclidean"
            )
        ),
        3,
    )
    davies_bouldin = round(
        davies_bouldin_score(latent_embeddings, cluster.cluster_labels_per_cluster), 3
    )
    calinksi_harabasz_score = round(
        calinski_harabasz_score(latent_embeddings, cluster.cluster_labels_per_cluster), 3
    )
    predicted_labels = cluster.cluster_labels_per_cluster

    purity_political, external_metrics_political = label_percentages(
        predicted_labels=predicted_labels, true_labels=true_labels_party
    )

    # purity_socioecon, external_metrics = label_percentages(
    #     predicted_labels=predicted_labels, true_labels=true_labels_socioecon
    # )

    purity_edulevel, external_metrics_edulevel = label_percentages(
        predicted_labels=predicted_labels, true_labels=true_labels_edulevel
    )
    return ClusteringPerformance(
        clustering_method=cluster.cluster_method,
        n_clusters=cluster.n_clusters,
        purity_political=purity_political,
        # purity_socioecon= purity_socioecon,
        purity_edulevel=purity_edulevel,
        silhouette_score=silhouette,
        predicted_labels=cluster.cluster_labels_per_cluster,
        num_cluster_score=cluster.score,
        davies_bouldin_score=davies_bouldin,
        calinksi_harabasz_score=calinksi_harabasz_score,
        inter_method_metrics_political=external_metrics_political,
        inter_method_metrics_edulevel=external_metrics_edulevel,
    )


# External Validation Metrics (Many-to-One)#
def label_percentages(
    *,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[float, dict[int, dict[str, float]]]:
    """Compute external validation metrics and combine with each clustering instance."""
    if len(predicted_labels) != len(true_labels):
        raise ValueError("The length of the predicted labels and true labels should be the same.")
    overall_purity = 0
    unique_labels = np.unique(predicted_labels).tolist()
    label_summarisation_dict = {label: {} for label in unique_labels}
    for label in unique_labels:
        stats_dict = {}
        percentages_dict = {}
        idx_list = []
        total_label_count = 0
        for idx, pred_label in np.ndenumerate(predicted_labels):
            if pred_label == label:
                stats_dict[true_labels[idx]] = stats_dict.get(true_labels[idx], 0) + 1

                idx_list.append(idx)
        total_label_count = sum(stats_dict.values())
        for key, value in stats_dict.items():
            percentages_dict[key] = round(value / total_label_count, 3)

        # TODO save id per item

        majority_label = max(stats_dict.items(), key=lambda x: x[1])[0]
        total_label_count = sum(stats_dict.values())

        cluster_purity_score = round(stats_dict[majority_label] / total_label_count, 3)
        label_summarisation_dict[label]["cluster_purity_score"] = cluster_purity_score
        label_summarisation_dict[label]["majority_label"] = majority_label
        label_summarisation_dict[label]["total_n_per_cluster"] = stats_dict
        label_summarisation_dict[label]["label_count"] = total_label_count
        label_summarisation_dict[label]["label_percentages"] = percentages_dict
        label_summarisation_dict[label]["index_list"] = idx_list
        overall_purity = cluster_purity_score
        #  (
        #     label_summarisation_dict[label]["label_count"] / len(predicted_labels)
        # )
    overall_purity = round(overall_purity, 3)

    return (overall_purity, label_summarisation_dict)
