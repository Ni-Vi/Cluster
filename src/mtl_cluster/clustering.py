import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import umap
import wandb
from loguru import logger
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

from mtl_cluster.cluster_related.metrics.compute_n_clusters import (
    agglomerative_clusters,
    gaussian_mixture_sklearn,
    hdbscan_clusters,
    kmeans_silhouette_clusters,
)
from mtl_cluster.cluster_related.metrics.internal_metrics import (
    ClusteringPerformance,
    cluster_numbers_validation_scores,
)
from mtl_cluster.cluster_related.modules.extract_latent_embeddings import (
    datamodule_to_embeddings_val_custom_model,
    datamodule_to_embeddings_val_pt_model,
)
from mtl_cluster.cluster_related.modules.table_concatenation_modification import (
    cluster_labels_anon,
    cluster_labels_babe,
    cluster_labels_gwsd,
    cluster_labels_mbic,
    concatenate_table_to_annotator_info_anon,
    concatenate_table_to_annotator_info_babe,
    concatenate_table_to_annotator_info_gwsd,
    concatenate_table_to_annotator_info_mbic,
    create_tables_from_cluster_metrics,
)
from mtl_cluster.data.datamodule import MTLDataModule
from mtl_cluster.data.datasets.constants import DatasetName
from mtl_cluster.lightning_modules.cluster import ClusterLightningModule
from mtl_cluster.modules.tokenise_split.tokenisers import (
    AnnotatorTokenizer,
)
from mtl_cluster.train_encoder_only_model import (
    DECODER_DATASET_PATH,
    ENCODER_DATASET_PATH,
    PRETRAINED_ENCODER_DECODER_DATASET_PATH,
    create_model as create_pt_model,
    create_pretrained_tokenizer,
    load_annotator_vocab,
)
from mtl_cluster.train_model import create_model as create_custom_model


SEED = 1000


def clustering_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clustering annotators.")
    parser.add_argument("--dataset_name", type=str, default="GWSD")
    parser.add_argument("--model", type=str, default="encoder_encoder")

    # gmm, kmeans, hdbscan, agglomerative
    parser.add_argument("--clustering_method", type=str, default="kmeans")
    parser.add_argument("--num_clusters", type=int, default=18)
    parser.add_argument("--hdbscan_eps", type=float, default=0.05)
    parser.add_argument("--hdbscan_min_samples", type=int, default=5)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5)

    parser.add_argument("--agglomerative_n_clusters", type=int, default=12)
    parser.add_argument("--agglomerative_distance_threshold", type=float, default=0.5)

    # umap, pca, mu_viswanath, none
    parser.add_argument("--dimensionality_reduction", type=str, default="umap")
    parser.add_argument("--normalize_after_reduction", type=bool, default=True)

    parser.add_argument("--umap_n_neighbors", type=int, default=100)
    parser.add_argument("--umap_min_dist", type=float, default=1)
    parser.add_argument("--umap_n_components", type=int, default=2)

    parser.add_argument("--pca_n_components", type=int, default=2)

    return parser.parse_args()


def get_args_from_run_id(run_id: str) -> argparse.Namespace:
    """Get the checkpoint path from the run id."""
    run = wandb.Api().run(f"lewidi/cluster/{run_id}")
    return argparse.Namespace(**run.config)


def determine_run_id(*, dataset: str, model: str) -> str:
    """Determine the WandB run ID we need from the pieces."""
    run_id_list = {
        DatasetName.anon: {
            "decoder_only": "bf3dpjvo",
            "encoder_decoder_pretrained": "ujv0fl21",
            "classifier": "lo0rtkd5",
            "cross_attention_pooled": "fsgpcef9",
            "encoder_encoder": "bfywnrb0",
            "cross_attention_unpooled": "oalt66qy",
        },
        DatasetName.babe: {
            "decoder_only": "wzcwjqh4",
            "encoder_decoder_pretrained": "yw7r7xua",
            "classifier": "w797j0wx",
            "cross_attention_pooled": "jq1t7akq",
            "encoder_encoder": "0omwx1jz",
            "cross_attention_unpooled": "oe45csj9",
        },
        DatasetName.gwsd: {
            "decoder_only": "09e6u77s",
            "encoder_decoder_pretrained": "9ivi67ue",
            "classifier": "lj1l49zm",
            "cross_attention_pooled": "1yhurzw1",
            "encoder_encoder": "lcwh0n6r",
            "cross_attention_unpooled": "vprdvn9z",
        },
        DatasetName.mbic: {
            "decoder_only": "0p6zke7w",
            "encoder_decoder_pretrained": "j7hro4ja",
            "classifier": "tzpkf36s",
            "cross_attention_pooled": "wuo9j1x0",
            "encoder_encoder": "2nlad5je",
            "cross_attention_unpooled": "mqn5aeo9",
        },
    }
    return run_id_list[DatasetName(dataset)][model]


def get_checkpoint_file_path(namespace: argparse.Namespace) -> Path:
    """Get the checkpoint path from the namespace."""
    root = Path(namespace.checkpoint_path)
    assert root.exists()
    assert root.is_dir()
    # Get the checkpoint file
    paths = list(root.glob("*.ckpt"))

    if len(paths) > 1:
        raise ValueError("There are more than one checkpoint file in the directory.")
    if not paths:
        raise FileNotFoundError("There are no checkpoint files in the directory.")

    return paths[0]


DATASETPATH = Path("storage/data/model_data/")
# CKPT = Path("src/mtl_cluster/checkpoints/Anon/2024-01-29_12-25-36/epoch=12-val_loss=0.53-val_acc=0.771230.ckpt")


def create_tables_from_clusters(
    dataset_name: DatasetName,
    embeddings_annotator_info_table: pl.DataFrame,
    cluster_performance: ClusteringPerformance,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and return tables from clusters."""
    predicted_cluster_annotator_info_table = embeddings_annotator_info_table.with_columns(
        pl.Series(name="clustering_predictions", values=cluster_performance.predicted_labels)
    )
    predicted_cluster_annotator_info_table = pd.DataFrame(
        predicted_cluster_annotator_info_table.to_dict(as_series=False)
    )
    if dataset_name == DatasetName.gwsd:
        predicted_cluster_annotator_info_table[
            ["party", "education", "gender"]
        ] = predicted_cluster_annotator_info_table[["party", "education", "gender"]].map(
            lambda x: x.value
        )
    elif dataset_name == DatasetName.babe:
        predicted_cluster_annotator_info_table[
            ["education", "gender", "native_speaker"]
        ] = predicted_cluster_annotator_info_table[["education", "gender", "native_speaker"]].map(
            lambda x: x.value
        )

    elif dataset_name == DatasetName.anon:
        predicted_cluster_annotator_info_table[
            ["native_speaker", "education", "gender"]
        ] = predicted_cluster_annotator_info_table[["native_speaker", "education", "gender"]].map(
            lambda x: x.value
        )
        predicted_cluster_annotator_info_table[
            "political_ideology"
        ] = predicted_cluster_annotator_info_table["political_ideology"].apply(
            lambda x: "left" if x < -3 else ("right" if x > 3 else "center")
        )
    else:
        predicted_cluster_annotator_info_table[
            ["education", "gender", "native_speaker"]
        ] = predicted_cluster_annotator_info_table[["education", "gender", "native_speaker"]].map(
            lambda x: x.value
        )

    edulevel_metrics = create_tables_from_cluster_metrics(
        cluster_metrics=cluster_performance.inter_method_metrics_edulevel
    )
    political_metrics = create_tables_from_cluster_metrics(
        cluster_metrics=cluster_performance.inter_method_metrics_political
    )
    predicted_cluster_annotator_info_table = predicted_cluster_annotator_info_table.drop(
        "embedding_space", axis=1
    )
    annotator_info_table = pd.DataFrame(predicted_cluster_annotator_info_table)
    edulevel_metrics_table = pd.DataFrame(edulevel_metrics.to_dict(as_series=False))
    political_metrics_table = pd.DataFrame(political_metrics.to_dict(as_series=False))

    # Figure out whether or not the clustering can go beyond just the text and/or the annotators
    num_sentences_per_cluster = []
    num_annotators_per_cluster = []
    length_of_cluster = []
    for _, group_df in annotator_info_table.groupby("clustering_predictions"):
        length_of_cluster.append(len(group_df))
        num_sentences_per_cluster.append(len(set(group_df["text"])))
        num_annotators_per_cluster.append(len(set(group_df["annotator_ids"])))

    edulevel_metrics_table.insert(3, "length_of_clusters", length_of_cluster, True)
    edulevel_metrics_table.insert(3, "unique_sentences", num_sentences_per_cluster, True)
    edulevel_metrics_table.insert(3, "unique_annotators", num_annotators_per_cluster, True)

    political_metrics_table.insert(3, "length_of_clusters", length_of_cluster, True)
    political_metrics_table.insert(3, "unique_sentences", num_sentences_per_cluster, True)
    political_metrics_table.insert(3, "unique_annotators", num_annotators_per_cluster, True)
    return annotator_info_table, edulevel_metrics_table, political_metrics_table


def look_inside_the_clusters(annotator_info: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return the indices and whats inside the goddamn cluster work dammit."""
    group_by_cluster_idx = annotator_info.groupby("clustering_predictions")
    return {  # pyright: ignore[reportReturnType]
        cluster_idx: group_df[
            ["text", "annotator_ids", "annotation_ids", "clustering_predictions"]
        ]
        for cluster_idx, group_df in group_by_cluster_idx
    }


def look_inside_the_clusters_new(annotator_info: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return the indices and whats inside the goddamn cluster work dammit."""
    group_by_cluster_idx = annotator_info.groupby("clustering_predictions")
    return {  # pyright: ignore[reportReturnType]
        cluster_idx: group_df[
            ["text", "annotator_ids", "clustering_predictions", "political_orientation"]
        ]
        for cluster_idx, group_df in group_by_cluster_idx
    }


def send_cluster_performance_to_wandb(
    model_run_id: str,
    cluster_performance: ClusteringPerformance,
    model_args: argparse.Namespace,
    cluster_args: argparse.Namespace,
    percentage_clusters_with_unique_sentences: float,
    percentage_clusters_with_multiple_annotators: float,
) -> None:
    """Send the performance to wandb to track is all."""
    wandb.init(
        project="cluster-performance",
        entity="lewidi",
        config={
            "model_run_id": model_run_id,
            **vars(model_args),
            **vars(cluster_args),
        },
    )
    assert wandb.run is not None
    logger.info("Logging the metrics to wandb.")
    wandb.run.log(
        {
            "n_clusters": cluster_performance.n_clusters,
            "purity_political": cluster_performance.purity_political,
            "purity_edulevel": cluster_performance.purity_edulevel,
            "silhouette_score": cluster_performance.silhouette_score,
            "davies_bouldin_score": cluster_performance.davies_bouldin_score,
            "calinksi_harabasz_score": cluster_performance.calinksi_harabasz_score,
            "percentage_clusters_with_unique_sentences": percentage_clusters_with_unique_sentences,
            "percentage_clusters_with_multiple_annotators": percentage_clusters_with_multiple_annotators,
            # Lasciate ogne speranza, voi ch'intrate
            "unique_percentages_combined": percentage_clusters_with_unique_sentences
            * percentage_clusters_with_multiple_annotators,
            "this_number_needs_to_be_1": percentage_clusters_with_unique_sentences
            * percentage_clusters_with_multiple_annotators
            * cluster_performance.silhouette_score,
        }
    )

    logger.info("Finishing the wandb run.")
    wandb.finish()


def dimensionality_reduction_mu_viswanath(
    embeddings: np.ndarray, *, num_components: int
) -> np.ndarray:
    """Reduce the dimensionality of the embeddings."""
    mean = embeddings.mean(0)
    num_components = embeddings.shape[-1] // 10
    #  shape of pca_components is (n_samples, n_components)
    pca_components = PCA(n_components=num_components).fit(embeddings - mean).components_
    preprocessed = embeddings - mean - ((embeddings @ pca_components.T) @ pca_components)

    return preprocessed


def main() -> None:
    """Veni, Vidi, Cluster.

    I came, I saw, I clustered annotators based on their annotation behaviour.- Ceasar (probably).
    """
    seed_everything(SEED)

    sweep_args = clustering_argparse()
    wandb_run_id = determine_run_id(dataset=sweep_args.dataset_name, model=sweep_args.model)

    model_args = get_args_from_run_id(wandb_run_id)

    logger.info("Creating wandb init")
    wandb.init(
        project="cluster-performance",
        entity="lewidi",
        config={
            "model_run_id": wandb_run_id,
            **vars(model_args),
            **vars(sweep_args),
        },
    )

    dataset = model_args.dataset_name
    ckpt = get_checkpoint_file_path(model_args)

    batch_size = 32
    if model_args.pretrained_model == "gpt2-large":
        dataset_path = DECODER_DATASET_PATH
    elif model_args.pretrained_model == "google/t5-v1_1-large":
        if model_args.is_train_encoder_model:
            dataset_path = PRETRAINED_ENCODER_DECODER_DATASET_PATH
        else:
            dataset_path = DATASETPATH
    else:
        dataset_path = ENCODER_DATASET_PATH

    logger.info("Loading the datamodule")
    datamodule = MTLDataModule(
        num_workers=17,
        datasetname=model_args.dataset_name,
        datapath=dataset_path,
        batch_size=batch_size,
        dataloader_kwargs={},
    )
    datamodule.setup("fit")

    unique_ann__metadatapath = DATASETPATH.joinpath(
        "unique_annotators/" + dataset + "_annotator_metadata.pkl"
    )
    annotator_info = pickle.load(unique_ann__metadatapath.open("rb"))

    logger.info("Creating the model")
    if model_args.is_train_encoder_model:
        lightning_module = ClusterLightningModule.load_from_checkpoint(
            ckpt, model=create_pt_model(model_args).model, map_location="cpu"
        )
        tokenizer = create_pretrained_tokenizer(
            model_args.pretrained_model, DatasetName(model_args.dataset_name)
        )
        extracted_embeddings_val = datamodule_to_embeddings_val_pt_model(
            datamodule, lightning_module, pretrained_tokenizer=tokenizer
        )
    else:
        lightning_module = ClusterLightningModule.load_from_checkpoint(
            ckpt, model=create_custom_model(model_args).model, map_location="cpu"
        )

        tokenizer = AnnotatorTokenizer(
            vocab=load_annotator_vocab(DatasetName(model_args.dataset_name))
        )
        extracted_embeddings_val = datamodule_to_embeddings_val_custom_model(
            datamodule,
            lightning_module,
            pretrained_model=model_args.pretrained_model,
            annotator_tokenizer=tokenizer,
        )

    latent_table = pl.from_arrow(extracted_embeddings_val.to_pyarrow_table())

    assert isinstance(latent_table, pl.DataFrame)
    # min_n_clusters = 5
    # max_n_clusters = 20
    if dataset == DatasetName.gwsd.value:
        embeddings_annotator_info_table = concatenate_table_to_annotator_info_gwsd(
            latent_table=latent_table, annotator_info=list(annotator_info.values())
        )
        latent_embeddings, political_labels, edulevel_labels = cluster_labels_gwsd(
            merged_table=embeddings_annotator_info_table
        )

    elif dataset == DatasetName.anon.value:
        embeddings_annotator_info_table = concatenate_table_to_annotator_info_anon(
            latent_table=latent_table, annotator_info=list(annotator_info.values())
        )
        latent_embeddings, political_labels, edulevel_labels = cluster_labels_anon(
            merged_table=embeddings_annotator_info_table
        )
    elif dataset == DatasetName.babe.value:
        embeddings_annotator_info_table = concatenate_table_to_annotator_info_babe(
            latent_table=latent_table, annotator_info=list(annotator_info.values())
        )
        latent_embeddings, political_labels, edulevel_labels = cluster_labels_babe(
            merged_table=embeddings_annotator_info_table
        )
        # min_n_clusters = 2
        # max_n_clusters = 6
    else:
        embeddings_annotator_info_table = concatenate_table_to_annotator_info_mbic(
            latent_table=latent_table, annotator_info=list(annotator_info.values())
        )
        latent_embeddings, political_labels, edulevel_labels = cluster_labels_mbic(
            merged_table=embeddings_annotator_info_table
        )

    if sweep_args.dimensionality_reduction == "umap":
        reduced_embeddings = umap.UMAP(
            n_neighbors=sweep_args.umap_n_neighbors,
            min_dist=sweep_args.umap_min_dist,
            n_components=sweep_args.umap_n_components,
        ).fit_transform(latent_embeddings)
        assert isinstance(reduced_embeddings, np.ndarray)

    elif sweep_args.dimensionality_reduction == "pca":
        reduced_embeddings = PCA(n_components=sweep_args.pca_n_components).fit_transform(
            latent_embeddings
        )
    elif sweep_args.dimensionality_reduction == "mu_viswanath":
        reduced_embeddings = dimensionality_reduction_mu_viswanath(
            latent_embeddings, num_components=sweep_args.pca_n_components
        )
    elif sweep_args.dimensionality_reduction == "none":
        reduced_embeddings = latent_embeddings

    if sweep_args.normalize_after_reduction:
        reduced_embeddings = Normalizer().fit_transform(reduced_embeddings)

    if sweep_args.clustering_method == "kmeans":
        num_clusters = kmeans_silhouette_clusters(
            min_n_cluster=sweep_args.num_clusters,
            max_n_cluster=sweep_args.num_clusters + 1,
            latent_embeddings=reduced_embeddings,
        )
    elif sweep_args.clustering_method == "gmm":
        num_clusters = gaussian_mixture_sklearn(
            min_n_cluster=sweep_args.num_clusters,
            max_n_cluster=sweep_args.num_clusters + 1,
            latent_embeddings=latent_embeddings,
        )
    elif sweep_args.clustering_method == "hdbscan":
        num_clusters = [
            hdbscan_clusters(
                latent_embeddings=latent_embeddings,
                min_samples=sweep_args.hdbscan_min_samples,
                min_cluster_size=sweep_args.hdbscan_min_cluster_size,
                eps=sweep_args.hdbscan_eps,
            )
        ]
    elif sweep_args.clustering_method == "agglomerative":
        num_clusters = [
            agglomerative_clusters(
                latent_embeddings=latent_embeddings,
                n_clusters=sweep_args.agglomerative_n_clusters,
                distance_threshold=sweep_args.agglomerative_distance_threshold,
            )
        ]
    else:
        raise ValueError("Clustering method not found.")

    try:
        cluster_performances = [
            cluster_numbers_validation_scores(
                cluster=cluster,
                latent_embeddings=reduced_embeddings,
                true_labels_party=political_labels,
                true_labels_edulevel=edulevel_labels,
            )
            for cluster in num_clusters
        ]
    except ValueError:
        assert wandb.run is not None
        logger.info("Logging the metrics to wandb.")
        wandb.run.log({"this_number_needs_to_be_1": 0})

        logger.info("Finishing the wandb run.")
        wandb.finish()
        return

    tables_from_performances = []

    for performance in cluster_performances:
        annotator_info, edulevel_metrics, political_metrics = create_tables_from_clusters(
            dataset_name=DatasetName(dataset),
            embeddings_annotator_info_table=embeddings_annotator_info_table,
            cluster_performance=performance,
        )

        tables_from_performances.append(
            (performance, annotator_info, edulevel_metrics, political_metrics)
        )

    for performance, _, _, political_metrics in tables_from_performances:
        # THIS NUMBER NEEDS TO BE BIG SO BIG OMG
        percentage_clusters_with_unique_sentences = sum(
            political_metrics["unique_sentences"] / political_metrics["length_of_clusters"]
        ) / len(political_metrics["unique_sentences"])
        # THIS NUMBER ALSO NEEDS TO BE HUGE OMG I'VE NEVER SEEN A NUMBER THIS BIG (but lower than 1)
        percentage_clusters_with_multiple_annotators = sum(
            political_metrics["unique_annotators"] / political_metrics["length_of_clusters"]
        ) / len(political_metrics["unique_annotators"])

        inside_the_clusters = look_inside_the_clusters(annotator_info)

        logger.info("Logging the metrics to wandb.")
        assert wandb.run is not None
        wandb.run.log(
            {
                "n_clusters": performance.n_clusters,
                "purity_political": performance.purity_political,
                "purity_edulevel": performance.purity_edulevel,
                "silhouette_score": performance.silhouette_score,
                "davies_bouldin_score": performance.davies_bouldin_score,
                "calinksi_harabasz_score": performance.calinksi_harabasz_score,
                "percentage_clusters_with_unique_sentences": percentage_clusters_with_unique_sentences,
                "percentage_clusters_with_multiple_annotators": percentage_clusters_with_multiple_annotators,
                # Lasciate ogne speranza, voi ch'intrate
                "unique_percentages_combined": percentage_clusters_with_unique_sentences
                * percentage_clusters_with_multiple_annotators,
                "this_number_needs_to_be_1": percentage_clusters_with_unique_sentences
                * percentage_clusters_with_multiple_annotators
                * performance.silhouette_score,
            }
        )

        logger.info("Finishing the wandb run.")
        wandb.finish()

    logger.info("Done.")

    # output_dir = Path("storage/data/cluster_metrics/").joinpath(dataset, model_args.model)
    # output_dir.mkdir(exist_ok=True, parents=True)
    # logger.info("Saving the cluster metrics to disk.")
    # date = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")

    # for performance in tqdm(cluster_numbers, desc="Starting a reduction method_dump"):
    #     output_file_prefix = f"{date}_{reduction}_{performance.clustering_method.value}_n={performance.n_clusters}_purity={performance.purity_political}_sil={performance.silhouette_score}"
    #     annotator_info, edulevel_metrics, political_metrics = create_tables_from_clusters(
    #         dataset_name=DatasetName(dataset),
    #         embeddings_annotator_info_table=embeddings_annotator_info_table,
    #         cluster_performance=performance,
    #     )

    #     unique_sentences_percentage = sum(political_metrics["unique_sentences"] <= 3) / len(
    #         political_metrics["unique_sentences"]
    #     )
    #     unique_annotators_percentage = sum(political_metrics["unique_annotators"] <= 1) / len(
    #         political_metrics["unique_annotators"]
    #     )

    #     # Figure out whether or not a performance is good.
    #     # 1. More than 50 clusters is bad.
    #     is_good_num_of_clusters = len(annotator_info["clustering_predictions"].unique()) <= 50
    #     # 2. More than 60% of the clusters have <= 3 unique sentences is bad.
    #     is_enough_sentences = unique_sentences_percentage <= 0.6
    #     # 3. More than 60% of the clusters have <= 1 unique annotators is bad.
    #     is_enough_annotators = unique_annotators_percentage <= 0.6

    #     is_good_performance = (
    #         is_good_num_of_clusters and is_enough_sentences and is_enough_annotators
    #     )
    #     if not is_good_performance:
    #         logger.warning(
    #             f"Performance is bad: {performance.clustering_method.value} {performance.n_clusters}"
    #         )
    #         continue

    #     annotator_info.to_csv(output_dir.joinpath(f"{output_file_prefix}_annotator_info.csv"))
    #     edulevel_metrics.to_csv(output_dir.joinpath(f"{output_file_prefix}_edulevel_metrics.csv"))
    #     political_metrics.to_csv(
    #         output_dir.joinpath(f"{output_file_prefix}_political_metrics.csv")
    #     )

    #     send_metrics_to_wandb(
    #         model_run_id=WANDB_RUN_ID,
    #         dataset_name=DatasetName(dataset),
    #         embeddings_annotator_info_table=embeddings_annotator_info_table,
    #         cluster_performance=metric,
    #         model_args=args,
    #     )


# def check_the_cluster_isnt_shit(
#     annotator_info_dataframe: pd.DataFrame,
#     *,
#     columns_we_care_about: list[str] = ["text", "annotator_ids"],
# ) -> bool:
#     # Group by clustering predictions
#     grouped_by_clustering_predictions = {}

#     for cluster_idx, group_df in annotator_info_dataframe.groupby("clustering_predictions"):
#         grouped_by_clustering_predictions[cluster_idx] = group_df[columns_we_care_about]

#     # Check 1: Check the number of unique sentences within a cluster.
#     num_sentences_per_cluster = {
#         cluster_idx: len(set(cluster_df["text"]))
#         for cluster_idx, cluster_df in grouped_by_clustering_predictions.items()
#     }

#     # Check 2: Number of unique annotators within a cluster
#     num_annotators_per_cluster = {
#         cluster_idx: len(set(cluster_df["annotator_ids"]))
#         for cluster_idx, cluster_df in grouped_by_clustering_predictions.items()
#     }


# #


# writer = SummaryWriter()
# writer.add_embedding(latent_embeddings, np.asarray(embeddings_annotator_info_table["annotation.ids"]))
# writer.close()
# %load_ext tensorboard
# %tensorboard --logdir=runs

if __name__ == "__main__":
    main()
