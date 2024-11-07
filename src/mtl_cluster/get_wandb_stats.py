from contextlib import suppress

import wandb
from rich.console import Console

from mtl_cluster.common.wand import get_best_run_id_from_sweep


SWEEP_IDS = (
    "mt7unnyh",  # Decoder Only  - Anon
    "941qcipw",  # Decoder Only - Babe
    "33fqags8",  # Decoder Only - GWSD
    "k1y0wchk",  # Decoder Only - MBIC
    "rtqco6l5",  # Encoder Decoder Pretrained - Anon
    "siiv7m0l",  # Encoder Decoder Pretrained - Babe
    "3nyc2udf",  # Encoder Decoder Pretrained - GWSD
    "9zxi7jwz",  # Encoder Decoder Pretrained - MBIC
    "r4a3sbxq",  # Classifier - Anon
    "3h8opzbm",  # Classifier - Babe
    "35k1g9vr",  # Classifier - GWSD
    "dkaarbek",  # Classifier - MBIC
    "64vmdp9v",  # Buffalo - Anon
    "vuqy95v5",  # Buffalo - Babe
    "6invxuva",  # Buffalo - GWSD
    "muist94w",  # Buffalo - MBIC
    "ccmolb18",  # Enc Enc - Anon
    "01y3zt6s",  # Enc Enc Babe
    "iq1mjo5j",  # Enc Enc - GWSD
    "dksk1yje",  # Enc Enc MBIC
    "o035uys1",  # Cross Attention - Anon
    "aa46nd9p",  # Cross Attention - Babe
    "4ydum9e6",  # Cross Attention - GWSD
    "nexcve1x",  # Cross Attention - MBIC
    # '6ctq5919' # Downsampled Cross Attention - GWSD
    # 'goeyp9sk' # Downsampled Cross Attention - Anon
    # 'ezqw2yy1' #  Cross Attention Pooled No Downsampling - GWSD
    # 'zjfcbmh6' # Cross Attention Pooled No Downsampling - Anon
)


def print_performance(run_id: str) -> str:
    """Get the evaluation performance from WandB for easy pasting in the paper."""
    # Load the run

    run = wandb.Api().run(f"lewidi/cluster/{run_id}")
    console = Console()
    # Print information about the run
    console.print("Run:", run.id)
    console.print("Name:", run.name)
    with suppress(KeyError):
        console.print("Training dataset:", run.config["training_data"])

    # Get all the success metrics per partition
    success_metrics = ("val_acc", "val/pairwise_similarity/mean", "val/pairwise_similarity/std")
    metrics_dict = {
        key: round(value, 2)
        for key, value in sorted(run.summary.items())
        if key in success_metrics
    }

    task_success = {
        "Accuracy": metrics_dict["val_acc"],
        "Mean similarity": metrics_dict["val/pairwise_similarity/mean"],
        "Std similarity": metrics_dict["val/pairwise_similarity/std"],
    }
    dataset = run._attrs["config"]["dataset"]
    try:
        model = run._attrs["config"]["model"]
    except:
        model = run._attrs["config"]["pretrained_model"]
    final_string = "{:.2f} & {:.2f} ({:.2f}) \\".format(
        task_success["Accuracy"], task_success["Mean similarity"], task_success["Std similarity"]
    )
    return dataset + " & " + model + " & " + " & " + final_string


def main() -> None:
    list_of_run_metrics = []

    for SWEEP_ID in SWEEP_IDS:
        run_id = get_best_run_id_from_sweep(SWEEP_ID)
        list_of_run_metrics.append(print_performance(run_id))

    console = Console()
    console.print(list_of_run_metrics)


if __name__ == "__main__":
    main()
