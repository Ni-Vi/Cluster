from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from pydantic import TypeAdapter

from mtl_cluster.data.datasets.anonymous_into_json import Annotator as AnonAnnotator
from mtl_cluster.data.datasets.babe_into_json import Annotator as BABEAnnotator
from mtl_cluster.data.datasets.gwsd_into_json import Annotator as GwsdAnnotator
from mtl_cluster.data.datasets.mbic_into_json import Annotator as MBICAnnotator


def concatenate_table_to_annotator_info_gwsd(
    *,
    latent_table: pl.DataFrame,
    annotator_info: list[GwsdAnnotator],
) -> pl.DataFrame:
    """Map the text to the annotator info."""
    annotator_info_dataframe = pl.from_records(
        TypeAdapter(list[GwsdAnnotator]).dump_python(annotator_info)
    )
    merged_table = latent_table.join(
        annotator_info_dataframe, how="inner", right_on="index", left_on="annotator_ids"
    )

    return merged_table


def concatenate_table_to_annotator_info_anon(
    *,
    latent_table: pl.DataFrame,
    annotator_info: list[AnonAnnotator],
) -> pl.DataFrame:
    """Map the text to the annotator info."""
    annotator_info_dataframe = pl.from_records(
        TypeAdapter(list[AnonAnnotator]).dump_python(annotator_info)
    )
    annotator_info_dataframe = annotator_info_dataframe.with_columns(
        pl.col("mturk_id").cast(pl.Int64)
    )
    merged_table = latent_table.join(
        annotator_info_dataframe, how="inner", right_on="mturk_id", left_on="annotator_ids"
    )

    return merged_table


def concatenate_table_to_annotator_info_mbic(
    *,
    latent_table: pl.DataFrame,
    annotator_info: list[MBICAnnotator],
) -> pl.DataFrame:
    """Map the text to the annotator info."""
    annotator_info_dataframe = pl.from_records(
        TypeAdapter(list[MBICAnnotator]).dump_python(annotator_info)
    )
    annotator_info_dataframe = annotator_info_dataframe.with_columns(
        pl.col("mturk_id").cast(pl.Int64)
    )
    merged_table = latent_table.join(
        annotator_info_dataframe, how="inner", right_on="mturk_id", left_on="annotator_ids"
    )

    return merged_table


def concatenate_table_to_annotator_info_babe(
    *,
    latent_table: pl.DataFrame,
    annotator_info: list[BABEAnnotator],
) -> pl.DataFrame:
    """Map the text to the annotator info."""
    annotator_info_dataframe = pl.from_records(
        TypeAdapter(list[BABEAnnotator]).dump_python(annotator_info)
    )
    merged_table = latent_table.join(
        annotator_info_dataframe, how="inner", right_on="annotator_id", left_on="annotator_ids"
    )
    merged_table = merged_table.drop_nulls()

    return merged_table


def cluster_labels_anon(merged_table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the true labels for the anon dataset."""
    latent_embeddings = np.array(merged_table["embedding_space"].to_list())
    true_labels_party = (
        merged_table["political_ideology"]
        .map_elements(lambda x: "left" if x < -3 else ("right" if x > 3 else "center"))
        .to_numpy()
    )
    true_labels_edulevel = merged_table["education"].to_numpy()

    return latent_embeddings, true_labels_party, true_labels_edulevel


def cluster_labels_gwsd(merged_table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the true labels for the gwsd dataset."""
    latent_embeddings = np.array(merged_table["embedding_space"].to_list())
    # TODO: Check this
    true_labels_party = merged_table["party"].map_elements(lambda x: x.name).to_numpy()
    true_labels_edulevel = merged_table["education"].to_numpy()

    return latent_embeddings, true_labels_party, true_labels_edulevel


def cluster_labels_babe(merged_table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the true labels for the anon dataset."""
    latent_embeddings = np.array(merged_table["embedding_space"].to_list())
    true_labels_party = (
        merged_table["political_ideology"]
        .map_elements(lambda x: "left" if x < -3 else ("right" if x > 3 else "center"))
        .to_numpy()
    )
    true_labels_edulevel = merged_table["education"].to_numpy()

    return latent_embeddings, true_labels_party, true_labels_edulevel


def cluster_labels_mbic(merged_table: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the true labels for the anon dataset."""
    latent_embeddings = np.array(merged_table["embedding_space"].to_list())
    true_labels_party = (
        merged_table["political_ideology"]
        .map_elements(lambda x: "left" if x < -3 else ("right" if x > 3 else "center"))
        .to_numpy()
    )
    true_labels_edulevel = merged_table["education"].to_numpy()

    return latent_embeddings, true_labels_party, true_labels_edulevel


def create_tables_from_cluster_metrics(
    *, cluster_metrics: dict[int, dict[str, Any]]
) -> pl.DataFrame:
    """Create tables from cluster metrics."""
    table_clusters = []

    for cluster_idx, cluster_met in cluster_metrics.items():
        # Note: this is messy and  hate i=t==t,=. I will fix it later. Promise. maybe,.
        counts = {
            f"{name}_count": count
            for name, count in cluster_met["total_n_per_cluster"].items()
            if isinstance(name, str)
        }
        more_counts = {
            f"{name.value}_count": count
            for name, count in cluster_met["total_n_per_cluster"].items()
            if isinstance(name, Enum)
        }
        percentages = {
            f"{name}_percentage": count
            for name, count in cluster_met["label_percentages"].items()
            if isinstance(name, str)
        }
        more_percentages = {
            f"{name.value}_percentage": count
            for name, count in cluster_met["label_percentages"].items()
            if isinstance(name, Enum)
        }

        majority_label = cluster_met["majority_label"]
        if isinstance(majority_label, Enum):
            majority_label = majority_label.value

        table_clusters.append(
            {
                "cluster_number": cluster_idx,
                "purity": cluster_met["cluster_purity_score"],
                "majority_label": majority_label,
                "label_count": cluster_met["label_count"],
                **counts,
                **more_counts,
                **percentages,
                **more_percentages,
            }
        )

    return pl.from_dicts(table_clusters)
