import torch
from loguru import logger
from rich.console import Console
from torchmetrics.functional.classification import (
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from mtl_cluster.clustering import (
    DATASETPATH,
    DECODER_DATASET_PATH,
    ENCODER_DATASET_PATH,
    PRETRAINED_ENCODER_DECODER_DATASET_PATH,
    ClusterLightningModule,
    MTLDataModule,
    create_custom_model,
    create_pt_model,
    determine_run_id,
    get_args_from_run_id,
    get_checkpoint_file_path,
)


IGNORE_INDEX = -100
console = Console()


def get_model_f1_score(dataset_name: str, model: str):
    """Get the model F1 score from the model."""
    wandb_run_id = determine_run_id(dataset=dataset_name, model=model)
    # Get the config from the run
    model_args = get_args_from_run_id(wandb_run_id)

    # Load the data
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

    datamodule = MTLDataModule(
        num_workers=17,
        datasetname=model_args.dataset_name,
        datapath=dataset_path,
        batch_size=batch_size,
        dataloader_kwargs={},
    )
    datamodule.setup("fit")

    # Load the model
    logger.info("Creating the model")
    if model_args.is_train_encoder_model:
        lightning_module = ClusterLightningModule.load_from_checkpoint(
            ckpt, model=create_pt_model(model_args).model, map_location="cpu"
        )
    else:
        lightning_module = ClusterLightningModule.load_from_checkpoint(
            ckpt, model=create_custom_model(model_args).model, map_location="cpu"
        )

    # Throw everything into the model and get all th targets and predictions
    predictions = []
    targets = []
    with torch.inference_mode():
        for batch in datamodule.val_dataloader():
            predicted_labels = lightning_module.forward(batch)

            if lightning_module.model_is_pooling:
                annotations, _ = torch.mode(
                    torch.where(
                        batch.annotations == IGNORE_INDEX,
                        torch.randint_like(batch.annotations, low=10, high=10000),
                        batch.annotations,
                    ),
                    dim=1,
                )
            else:
                annotations = batch.annotations

            targets.append(annotations.flatten())
            predictions.append(predicted_labels.reshape(-1, predicted_labels.size(-1)))

    # Throw it all into F1 and get the score.
    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    logger.info("Calculating the F1 score")
    f1_score = multiclass_f1_score(
        predictions_tensor, targets_tensor, num_classes=5, ignore_index=IGNORE_INDEX
    )
    precision = multiclass_precision(
        predictions_tensor, targets_tensor, num_classes=5, ignore_index=IGNORE_INDEX
    )
    recall = multiclass_recall(
        predictions_tensor, targets_tensor, num_classes=5, ignore_index=IGNORE_INDEX
    )

    return round(float(f1_score), 2), round(float(precision), 2), round(float(recall), 2)
    # Done.


if __name__ == "__main__":
    results = []

    datasets = (
        "GWSD",
        # "MBIC",
    )
    models = [
        # "cross_attention_unpooled",
        # "cross_attention_pooled",
        # "encoder_encoder",
        "classifier",
        # "decoder_only",
        "encoder_decoder_pretrained",
    ]
    for dataset in datasets:
        for model in models:
            results.append((dataset, model, *get_model_f1_score(dataset, model)))

    console.rule()

    for dataset, model, f1, precision, recall in results:
        console.print(f"{dataset} & {model} & {f1} & {precision} & {recall}")

    console.rule()
