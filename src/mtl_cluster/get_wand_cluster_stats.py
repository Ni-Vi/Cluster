from contextlib import suppress

import wandb
from rich.console import Console


RUN_IDS = (
    # MBIC classifier pca kmeans
    "fymdnxtn",
    # MBIC classifier umap kmeans
    "i2c7iasi",
    # GWSD classifier pca kmeans
    "3tbhq5sj",
    # GWSD classifier umap kmeans
    "v099bj92",
    # MBIC decoder_only none kmeans
    "09s3ly1k",
    # MBIC encoder_decoder_pretrained none kmeans
    "i0ui0pen",
    # MBIC classifier none kmeans
    "41zs7nuk",
    # MBIC cross_attention_pooled none kmeans
    "0091kxlb",
    # MBIC encoder_encoder none kmeans
    "0gy4ha85",
    # MBIC cross_attention_unpooled none kmeans
    "0l2eo0nk",
    #  GWSD decoder_only none kmeans
    "grdxq3h6",
    # GWSD encoder_decoder_pretrained none kmeans
    "4bnooa3d",
    # GWSD classifier none kmeans
    "072kamtn",
    # GWSD cross_attention_pooled none kmeans
    "3aa8nqk4",
    # GWSD encoder_decoder_pretrained none kmeans
    "4bnooa3d",
    # GWSD encoder_encoder none kmeans
    "z777ct60",
    # GWSD cross_attention_unpooled none kmeans
    "0ifo0lsw",
    # # Anon encoder_encoder pca kmeans
    # "044xb4iu",
    # # Anon encoder_encoder umap kmeans
    # "69uhpq21",
    # # Anon cross_attention_unpooled pca kmeans
    # "6be78bmf",
    # # Anon cross_attention_unpooled umap kmeans
    # "gmr5wkfw",  # ej2kcpmv
    # # GWSD decoder_only pca kmeans
    # "rd6u1ep0",
    # # GWSD decoder_only umap kmeans
    # "5dvn46qf",
    # # GWSD encoder_decoder_pretrained pca kmeans
    # "2eephxsc",
    # # GWSD encoder_decoder_pretrained umap kmeans
    # "gf76qnqi",
    # # GWSD cross_attention_pooled pca kmeans
    # "et291b8n",
    # # GWSD cross_attention_pooled umap kmeans
    # "8lbotchh",
    # # GWSD encoder_encoder pca kmeans
    # "03hh9ikh",
    # # GWSD encoder_encoder umap kmeans
    # "w3qasx7u",
    # # GWSD cross_attention_unpooled pca kmeans
    # "eq2758r8",
    # # GWSD cross_attention_unpooled umap kmeans
    # "k6t731wp",
)


def print_performance(run_id: str) -> str:
    """Get the evaluation performance from WandB for easy pasting in the paper."""
    # Load the run

    run = wandb.Api().run(f"lewidi/cluster-performance/{run_id}")
    console = Console()
    # Print information about the run
    console.print("Run:", run.id)
    console.print("Name:", run.name)
    with suppress(KeyError):
        console.print("Training dataset:", run.config["training_data"])

    # Get all the success metrics per partition
    success_metrics = (
        "n_clusters",
        "davies_bouldin_score",
        "silhouette_score",
        "purity_political",
        "purity_edulevel",
    )
    metrics_dict = {
        key: round(value, 2)
        for key, value in sorted(run.summary.items())
        if key in success_metrics
    }

    task_success = {
        "n_clusters": metrics_dict["n_clusters"],
        "davies_bouldin_score": metrics_dict["davies_bouldin_score"],
        "silhouette_score": metrics_dict["silhouette_score"],
        "purity_political": metrics_dict["purity_political"],
        "purity_edulevel": metrics_dict["purity_edulevel"],
    }

    dimensionality_reduction = run._attrs["config"]["dimensionality_reduction"]
    dataset = run._attrs["config"]["dataset"]
    try:
        model = run._attrs["config"]["model"]
    except:
        model = run._attrs["config"]["pretrained_model"]
    final_string = "{:.2f} & {:.2f}  &  {:.2f}  & {:.2f} & {:.2f}\\".format(
        task_success["n_clusters"],
        task_success["davies_bouldin_score"],
        task_success["silhouette_score"],
        task_success["purity_political"],
        task_success["purity_edulevel"],
    )
    return dataset + "" + model + "" + dimensionality_reduction + " & " + " & " + final_string


def main() -> None:
    list_of_run_metrics = []

    for run_id in RUN_IDS:
        list_of_run_metrics.append(print_performance(run_id))

    console = Console()
    console.print(list_of_run_metrics)


if __name__ == "__main__":
    main()
