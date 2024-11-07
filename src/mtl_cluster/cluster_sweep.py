from time import sleep

import wandb


DATASETS = [
    "MBIC",
    "GWSD",
    # "Babe",
    # "Anon",
]
MODELS = [
    "decoder_only",
    "encoder_decoder_pretrained",
    "classifier",
    "cross_attention_pooled",
    "encoder_encoder",
    "cross_attention_unpooled",
]
DIMENSIONALITY_REDUCTION = [
    # "pca",
    # "umap",
    "none",
    # "mu_viswanath",
]
CLUSTERING_ALGORITHMS = [
    # "kmeans",
    "hdbscan",
    # "gmm",
    # "agglomerative",
]

DATASET_CLUSTER_RANGES = {
    "MBIC": (5, 20),
    "Babe": (2, 6),
    "Anon": (5, 20),
    "GWSD": (5, 20),
}


HDBSCAN_OPTIONS = {
    "hdbscan_eps": {
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "hdbscan_min_samples": {
        "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
    },
    "hdbscan_min_cluster_size": {
        "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
    },
}

PCA_OPTIONS = {
    "pca_n_components": {
        "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40],
    }
}

AGGLOMERATIVE_OPTIONS = {
    "agglomerative_n_clusters": {
        "distribution": "int_uniform",
        "min": 3,
        "max": 30,
    },
    "agglomerative_distance_threshold": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 1.0,
    },
}

UMAP_OPTIONS = {
    "umap_n_components": {
        "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
    },
    "umap_n_neighbors": {
        "values": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100],
    },
    "umap_min_dist": {
        "values": [
            0.0,
            0.001,
            0.025,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    },
}


def main() -> None:
    name_template = "{dataset} {model} {reduction} {cluster}"
    config = {
        "command": ["${env}", "python", "${program}", "${args}"],
        "entity": "lewidi",
        "project": "cluster-performance",
        "program": "src/mtl_cluster/clustering.py",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "this_number_needs_to_be_1", "target": 1},
        "parameters": {
            "normalize_after_reduction": {"values": [True, False]},
        },
    }

    all_sweep_configs = []

    for dataset in DATASETS:
        for model in MODELS:
            for reduction in DIMENSIONALITY_REDUCTION:
                for cluster in CLUSTERING_ALGORITHMS:
                    sweep_name = name_template.format(
                        dataset=dataset, model=model, reduction=reduction, cluster=cluster
                    )
                    cluster_options = {}
                    num_clusters = {}
                    dimensionality_reduction_options = {}

                    if cluster == "hdbscan":
                        cluster_options = HDBSCAN_OPTIONS
                    elif cluster == "agglomerative":
                        cluster_options = AGGLOMERATIVE_OPTIONS

                    if reduction in ("pca", "mu_viswanath"):
                        dimensionality_reduction_options = PCA_OPTIONS
                    elif reduction == "umap":
                        dimensionality_reduction_options = UMAP_OPTIONS

                    if cluster in ("gmm", "kmeans"):
                        num_clusters = {
                            "num_clusters": {
                                "values": list(
                                    range(
                                        DATASET_CLUSTER_RANGES[dataset][0],
                                        DATASET_CLUSTER_RANGES[dataset][1],
                                    )
                                ),
                            }
                        }

                    sweep_config = {
                        **config,
                        "name": sweep_name,
                        "parameters": {
                            **config["parameters"],
                            "dataset_name": {"value": dataset},
                            "model": {"value": model},
                            "dimensionality_reduction": {"value": reduction},
                            "clustering_method": {"value": cluster},
                            **cluster_options,
                            **dimensionality_reduction_options,
                            **num_clusters,
                        },
                    }
                    all_sweep_configs.append(sweep_config)

    sweep_ids = []

    for sweep_config in all_sweep_configs:
        sweep_id = wandb.sweep(sweep=sweep_config, entity="lewidi", project="cluster-performance")
        sweep_ids.append(sweep_id)
        sleep(1)

    for sweep_config, sweep_id in zip(all_sweep_configs, sweep_ids, strict=True):
        print(f"# {sweep_config['name']}")
        print(f'"{sweep_id}"')


if __name__ == "__main__":
    main()
