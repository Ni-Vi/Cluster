import argparse
import json
import os
import pickle
from functools import lru_cache, partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from rich.pretty import pprint as rich_print
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    get_constant_schedule_with_warmup,
)

from mtl_cluster.common.wand import get_args_from_run_id, get_best_run_id_from_sweep
from mtl_cluster.data.datamodule import MTLDataModule
from mtl_cluster.data.dataset_instance import ModelInstance
from mtl_cluster.data.datasets.constants import DatasetName
from mtl_cluster.data.encoder_only_dataset import (
    prepare_instances_for_decoder_only_model,
    prepare_instances_for_encoder_decoder_pretrained_model,
    prepare_instances_for_encoder_only_model,
)
from mtl_cluster.lightning_modules.cluster import ClusterLightningModule
from mtl_cluster.models.models import (
    DecoderOnlyModel,
    EncoderDecoderPretrainedModel,
    SequenceClassificationModel,
)
from mtl_cluster.modules.tokenise_split.tokenisers import (
    AnnotationTokenizer,
    annotation_label_vocabulary_build,
    get_annotation_vocab_size,
)
from mtl_cluster.preprocessing_data import TOKENIZER_PATH, TRAIN_PATH, VAL_PATH
from mtl_cluster.train_model import SEED, create_checkpoint_path, get_padding_tokens


ENABLE_CHECKPOINTING = os.environ.get("ENABLE_CHECKPOINTING", "False").lower() in {"true", "1"}
SWEEP_ID = os.environ.get("SWEEP_ID", "rtqco6l5")
# SWEEP_ID = "dkaarbek"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DATA_PATH = Path("storage/data/model_data/")

UNIQUE_ANN_PATH = MODEL_DATA_PATH.joinpath("unique_annotators/")

# PRETRAINED_MODEL = "bert-large-uncased"
# PRETRAINED_MODEL = "roberta-large"
# PRETRAINED_MODEL : "gpt2-large"
# PRETRAINED_MODEL : "google/t5-v1_1-large"
ENCODER_DATASET_PATH = Path("storage/data/encoder_model_data/")
TRAIN_PATH_ENCODER_ONLY = ENCODER_DATASET_PATH.joinpath("train/")
VAL_PATH_ENCODER_ONLY = ENCODER_DATASET_PATH.joinpath("val/")

DECODER_DATASET_PATH = Path("storage/data/decoder_model_data/")
TRAIN_PATH_DECODER_ONLY = DECODER_DATASET_PATH.joinpath("train/")
VAL_PATH_DECODER_ONLY = DECODER_DATASET_PATH.joinpath("val/")

PRETRAINED_ENCODER_DECODER_DATASET_PATH = Path(
    "storage/data/pretrained_encoder_decoder_model_data/"
)
TRAIN_PATH_PRETRAINED_ENCODER_DECODER = PRETRAINED_ENCODER_DECODER_DATASET_PATH.joinpath("train/")
VAL_PATH_PRETRAINED_ENCODER_DECODER = PRETRAINED_ENCODER_DECODER_DATASET_PATH.joinpath("val/")


@lru_cache(maxsize=1)
def load_annotator_vocab(dataset_name: DatasetName) -> dict[str, int]:
    """Load the annotator vocab from the path."""
    annotator_vocab: dict[str, int] = json.loads(
        TOKENIZER_PATH.joinpath(f"{dataset_name.value}_annotator_vocab.json").read_text()
    )
    return annotator_vocab


def get_num_unique_annotators(dataset_name: DatasetName) -> int:
    """Get the number of unique annotators."""
    annotator_vocab = load_annotator_vocab(dataset_name)
    return len(annotator_vocab)


def create_pretrained_tokenizer(
    pretrained_model: str, dataset_name: DatasetName
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Create the tokenizer for the pretrained model."""
    pretrained_tokenizer_path = TOKENIZER_PATH.joinpath(
        f"{pretrained_model.replace('/', '-')}_{dataset_name.value}_tokenizer.pkl"
    )

    if pretrained_tokenizer_path.exists():
        logger.info("Loading the tokenizer from the path.")
        tokenizer = pickle.load(pretrained_tokenizer_path.open("rb"))  # noqa: S301
    else:
        logger.info("Loading the annotator vocab from the path.")
        annotator_vocab = load_annotator_vocab(dataset_name)
        logger.info("Creating the tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        annotator_tokens = sorted(annotator_vocab.keys())
        tokenizer.add_tokens(list(annotator_tokens))

        logger.info("Dumping the tokenizer")
        pickle.dump(tokenizer, pretrained_tokenizer_path.open("wb"))

    return tokenizer


def _argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="Babe")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--is_train_encoder_model", type=bool, default=True)

    if SWEEP_ID is not None:
        logger.info("Getting args from wandb sweep")
        best_run = get_best_run_id_from_sweep(SWEEP_ID)
        args = get_args_from_run_id(best_run)
    else:
        args = parser.parse_args()

    rich_print(args)
    return args


def create_model(args: argparse.Namespace) -> ClusterLightningModule:
    """Create the lightning module from the args for the classifier models."""
    dataset_name = DatasetName(args.dataset)
    annotation_tokenizer = AnnotationTokenizer(annotation_label_vocabulary_build())

    tokenizer = create_pretrained_tokenizer(args.pretrained_model, dataset_name)
    num_unique_annotators = get_num_unique_annotators(dataset_name)
    if args.pretrained_model == "gpt2-large":
        create_decoder_only_preprocessed_data(dataset_name, tokenizer, annotation_tokenizer)
        model = DecoderOnlyModel.from_pretrained(
            args.pretrained_model,
            unique_annotators=num_unique_annotators,
            annotation_vocab_size=len(annotation_tokenizer.vocab),
            separator_tokens=get_padding_tokens(args.pretrained_model),
        )
    elif args.pretrained_model == "google/t5-v1_1-large":
        create_encoder_decoder_preprocessed_data(dataset_name, tokenizer, annotation_tokenizer)
        model = EncoderDecoderPretrainedModel.from_pretrained(
            args.pretrained_model,
            unique_annotators=num_unique_annotators,
            annotation_vocab_size=len(annotation_tokenizer.vocab),
        )

    else:
        create_encoder_only_preprocessed_data(dataset_name, tokenizer, annotation_tokenizer)
        model = SequenceClassificationModel.from_pretrained(
            args.pretrained_model,
            unique_annotators=num_unique_annotators,
            annotation_vocab_size=len(annotation_tokenizer.vocab),
        )

    learning_rate_scheduler_partial = partial(
        get_constant_schedule_with_warmup, num_warmup_steps=args.num_warmup_steps
    )

    optimizer_partial_fn = partial(torch.optim.AdamW, lr=args.learning_rate, weight_decay=0.01)
    logger.info("Creating the lightning module")
    lightning_module = ClusterLightningModule(
        model=model,
        optimizer_partial_fn=optimizer_partial_fn,
        lr_scheduler_partial_fn=learning_rate_scheduler_partial,
        annotation_covab_size=get_annotation_vocab_size(),
    )

    return lightning_module


def load_model_instances(
    dataset_name: DatasetName,
) -> tuple[list[ModelInstance], list[ModelInstance]]:
    """Load the model instances from the pkl files."""
    train_path = TRAIN_PATH.joinpath(f"{dataset_name.value}_train_unencoded.pkl")
    val_path = VAL_PATH.joinpath(f"{dataset_name.value}_val_unencoded.pkl")
    train_model_instances = pickle.load(train_path.open("rb"))
    val_model_instances = pickle.load(val_path.open("rb"))

    return train_model_instances, val_model_instances


def create_decoder_only_preprocessed_data(
    dataset_name: DatasetName,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> None:
    train_model_instances, val_model_instances = load_model_instances(dataset_name)
    # removing the instance where the text is just the word "may"
    if dataset_name == DatasetName.babe:
        train_model_instances = [
            instance for instance in train_model_instances if instance.sentence != "may"
        ]

    preprocessed_train_instances = prepare_instances_for_decoder_only_model(
        train_model_instances, tokenizer, annotation_tokenizer
    )
    preprocessed_val_instances = prepare_instances_for_decoder_only_model(
        val_model_instances, tokenizer, annotation_tokenizer
    )

    TRAIN_PATH_DECODER_ONLY.mkdir(parents=True, exist_ok=True)
    VAL_PATH_DECODER_ONLY.mkdir(parents=True, exist_ok=True)

    with TRAIN_PATH_DECODER_ONLY.joinpath(f"{dataset_name.value}_separate_train.pkl").open(
        "wb"
    ) as f:
        pickle.dump(preprocessed_train_instances, f)

    with VAL_PATH_DECODER_ONLY.joinpath(f"{dataset_name.value}_separate_val.pkl").open("wb") as f:
        pickle.dump(preprocessed_val_instances, f)


def create_encoder_decoder_preprocessed_data(
    dataset_name: DatasetName,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> None:
    train_model_instances, val_model_instances = load_model_instances(dataset_name)
    # tokenizer is the same with the decoder model architecture
    preprocessed_train_instances = prepare_instances_for_encoder_decoder_pretrained_model(
        train_model_instances, tokenizer, annotation_tokenizer
    )
    preprocessed_val_instances = prepare_instances_for_encoder_decoder_pretrained_model(
        val_model_instances, tokenizer, annotation_tokenizer
    )

    TRAIN_PATH_PRETRAINED_ENCODER_DECODER.mkdir(parents=True, exist_ok=True)
    VAL_PATH_PRETRAINED_ENCODER_DECODER.mkdir(parents=True, exist_ok=True)

    with TRAIN_PATH_PRETRAINED_ENCODER_DECODER.joinpath(
        f"{dataset_name.value}_separate_train.pkl"
    ).open("wb") as f:
        pickle.dump(preprocessed_train_instances, f)

    with VAL_PATH_PRETRAINED_ENCODER_DECODER.joinpath(
        f"{dataset_name.value}_separate_val.pkl"
    ).open("wb") as f:
        pickle.dump(preprocessed_val_instances, f)


def create_encoder_only_preprocessed_data(
    dataset_name: DatasetName,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> None:
    train_model_instances, val_model_instances = load_model_instances(dataset_name)

    preprocessed_train_instances = prepare_instances_for_encoder_only_model(
        train_model_instances, tokenizer, annotation_tokenizer
    )
    preprocessed_val_instances = prepare_instances_for_encoder_only_model(
        val_model_instances, tokenizer, annotation_tokenizer
    )

    TRAIN_PATH_ENCODER_ONLY.mkdir(parents=True, exist_ok=True)
    VAL_PATH_ENCODER_ONLY.mkdir(parents=True, exist_ok=True)

    with TRAIN_PATH_ENCODER_ONLY.joinpath(f"{dataset_name.value}_separate_train.pkl").open(
        "wb"
    ) as f:
        pickle.dump(preprocessed_train_instances, f)

    with VAL_PATH_ENCODER_ONLY.joinpath(f"{dataset_name.value}_separate_val.pkl").open("wb") as f:
        pickle.dump(preprocessed_val_instances, f)


def main() -> None:
    seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    args = _argument_parser()

    dataset_name = DatasetName(args.dataset)

    if args.pretrained_model == "gpt2-large":
        dataset_path = DECODER_DATASET_PATH
        batch_size = 32
    elif args.pretrained_model == "google/t5-v1_1-large":
        dataset_path = PRETRAINED_ENCODER_DECODER_DATASET_PATH
        batch_size = 32
    else:
        dataset_path = ENCODER_DATASET_PATH
        batch_size = 32

    datamodule = MTLDataModule(
        num_workers=17,
        datasetname=dataset_name.value,
        datapath=dataset_path,
        batch_size=batch_size,
        dataloader_kwargs={},
    )

    lightning_module = create_model(args)

    model_checkpoint_path = create_checkpoint_path(dataset_name=dataset_name)

    pretrained_model_to_model = {
        "gpt2-large": "decoder_only",
        "google/t5-v1_1-large": "encoder_decoder",
        "bert-large-uncased": "classifier",
        "roberta-large": "classifier",
    }

    logger.info("Setting up wandb.")
    wandb_logger = WandbLogger(
        project="cluster",
        entity="lewidi",
        log_model=False,
        name=f"{pretrained_model_to_model[args.pretrained_model]} - {args.dataset}"
        if SWEEP_ID is None
        else None,
        group="Best Models" if SWEEP_ID is not None else None,
        config={
            **vars(args),
            "dataset_name": dataset_name.value,
            "model": pretrained_model_to_model[args.pretrained_model],
            "checkpoint_path": str(model_checkpoint_path),
        },
    )
    wandb_logger.watch(lightning_module, log="all", log_freq=17, log_graph=True)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=False,
        min_delta=0.001,
        mode="min",
        check_finite=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoint_path,
        filename="{epoch}-{val_loss:.2f}-{val_acc:2f}",
        monitor="val_loss",
        save_weights_only=True,
        save_top_k=1,
    )

    callbacks = [
        RichModelSummary(),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        early_stop_callback,
    ]
    if ENABLE_CHECKPOINTING:
        callbacks.append(checkpoint_callback)

    logger.info("Setting up trainer")
    trainer = pl.Trainer(
        max_epochs=500,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        gradient_clip_val=1,
        log_every_n_steps=10,
        # num_sanity_val_steps= 0,
        logger=wandb_logger,
    )

    logger.info("Starting the training...")
    trainer.fit(lightning_module, datamodule=datamodule)
    logger.info("Done.")


# Load the model instances

if __name__ == "__main__":
    main()
