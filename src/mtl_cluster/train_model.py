import argparse
import datetime
import os
import pickle
from functools import partial
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
from transformers import PreTrainedModel, get_constant_schedule_with_warmup
from transformers.models.t5.modeling_t5 import T5EncoderModel

from mtl_cluster.common.wand import get_args_from_run_id, get_best_run_id_from_sweep
from mtl_cluster.data.datamodule import MTLDataModule
from mtl_cluster.data.datasets.constants import DatasetName
from mtl_cluster.lightning_modules.cluster import ClusterLightningModule
from mtl_cluster.models.models import (
    CrossAttentionModel,
    CrossAttentionPooledModel,
    EncoderEncoderModel,
    EncoderEncoderPooledModel,
)
from mtl_cluster.modules.tokenise_split.tokenisers import TextTokenizer, get_annotation_vocab_size


SEED = 1000
ENABLE_CHECKPOINTING = os.environ.get("ENABLE_CHECKPOINTING", "True").lower() in {"true", "1"}
SWEEP_ID = os.environ.get("SWEEP_ID", "iq1mjo5j")
PRETRAINED_MODEL_DEFAULT = "google/t5-v1_1-large"


def arguments_parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--should_freeze_encoder", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--disable_transformer_ff_bias", type=int, default=0)
    parser.add_argument("--disable_projection_bias", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=540)
    parser.add_argument("--ann_dim_factor", type=int, default=1)
    # possible model args = "cross_attention_unpooled", "cross_attention_pooled", "encoder_encoder_unpooled", "encoder_encoder_pooled"
    parser.add_argument("--model", type=str, default="encoder_encoder")
    # possible dataset args = "Anon", "GWSD", "MBIC", Babe
    parser.add_argument("--dataset", type=str, default="MBIC")
    parser.add_argument("--pretrained_model", type=str, default=PRETRAINED_MODEL_DEFAULT)
    parser.add_argument("--downsample_num_layers", type=int, default=0)
    parser.add_argument("--is_train_encoder_model", type=bool, default=False)
    args = parser.parse_args()

    if SWEEP_ID is not None:
        logger.info("Getting args from wandb sweep")
        best_run = get_best_run_id_from_sweep(SWEEP_ID)
        args_from_run = get_args_from_run_id(best_run)
        args = argparse.Namespace(**{**vars(args), **vars(args_from_run)})

    rich_print(args)
    return args


CKPT = Path("src/mtl_cluster/checkpoints/")
DATASETPATH = Path("storage/data/model_data/")
CONFIGPATH = Path("config/")


def get_padding_tokens(pretrained_model: str) -> list[int]:
    """Gets the padding tokens for the model."""
    text_tokenizer = TextTokenizer(pretrained_model)
    padding_str = "//"
    padding_token_list = text_tokenizer.tokenizer.convert_tokens_to_ids(padding_str)
    if not isinstance(padding_token_list, list):
        padding_token_list = [padding_token_list]
    return padding_token_list


def create_checkpoint_path(dataset_name: DatasetName) -> Path:
    """Create a checkpoint path for the model."""
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = CKPT.joinpath(dataset_name.value).joinpath(timestamp)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    return checkpoint_path


def create_model(args: argparse.Namespace) -> ClusterLightningModule:
    """Create the lightning module from the args."""
    encoder = T5EncoderModel.from_pretrained(args.pretrained_model)

    assert isinstance(encoder, PreTrainedModel)

    if args.should_freeze_encoder == 1:
        logger.info("Freezing encoder")
        for param in encoder.parameters():
            param.requires_grad = False

    unique_ann_path = DATASETPATH.joinpath(
        "unique_annotators/" + args.dataset + "_annotator_metadata.pkl"
    )
    annotator_info = pickle.load(unique_ann_path.open("rb"))  # noqa: S301
    n_annotators = len(annotator_info)

    logger.info("Creating the model")
    if args.model in {"cross_attention_unpooled", "cross_attention"}:
        model = CrossAttentionModel(
            encoder=encoder,
            unique_annotators=n_annotators,
            annotation_vocab_size=get_annotation_vocab_size(),
            decoder_depth=args.decoder_depth,
            modalities=args.decoder_heads,
            downsample_num_layers=args.downsample_num_layers,
            dropout=args.dropout,
            text_dim=encoder.config.d_model,
            ann_dim=int(encoder.config.d_model // args.ann_dim_factor),
            disable_transformer_ff_bias=bool(args.disable_transformer_ff_bias),
            disable_projection_bias=bool(args.disable_projection_bias),
            padding_tokens=get_padding_tokens(args.pretrained_model),
        )

    elif args.model == "cross_attention_pooled":
        model = CrossAttentionPooledModel(
            encoder=encoder,
            unique_annotators=n_annotators,
            annotation_vocab_size=get_annotation_vocab_size(),
            decoder_depth=args.decoder_depth,
            modalities=args.decoder_heads,
            downsample_num_layers=args.downsample_num_layers,
            dropout=args.dropout,
            text_dim=encoder.config.d_model,
            ann_dim=int(encoder.config.d_model // args.ann_dim_factor),
            disable_transformer_ff_bias=bool(args.disable_transformer_ff_bias),
            disable_projection_bias=bool(args.disable_projection_bias),
            padding_tokens=get_padding_tokens(args.pretrained_model),
        )

    elif args.model in {"encoder_encoder_unpooled", "encoder_encoder"}:
        model = EncoderEncoderModel(
            encoder=encoder,
            unique_annotators=n_annotators,
            annotation_vocab_size=get_annotation_vocab_size(),
            decoder_depth=args.decoder_depth,
            modalities=args.decoder_heads,
            downsample_num_layers=args.downsample_num_layers,
            dropout=args.dropout,
            text_dim=encoder.config.d_model,
            ann_dim=int(encoder.config.d_model // args.ann_dim_factor),
            disable_transformer_ff_bias=bool(args.disable_transformer_ff_bias),
            disable_projection_bias=bool(args.disable_projection_bias),
            padding_tokens=get_padding_tokens(args.pretrained_model),
        )

    elif args.model == "encoder_encoder_pooled":
        model = EncoderEncoderPooledModel(
            encoder=encoder,
            unique_annotators=n_annotators,
            annotation_vocab_size=get_annotation_vocab_size(),
            decoder_depth=args.decoder_depth,
            modalities=args.decoder_heads,
            downsample_num_layers=args.downsample_num_layers,
            dropout=args.dropout,
            text_dim=encoder.config.d_model,
            ann_dim=int(encoder.config.d_model // args.ann_dim_factor),
            disable_transformer_ff_bias=bool(args.disable_transformer_ff_bias),
            disable_projection_bias=bool(args.disable_projection_bias),
            padding_tokens=get_padding_tokens(args.pretrained_model),
        )

    else:
        raise ValueError(f"Model not found: {args.model}")

    learning_rate_scheduler_partial = partial(
        get_constant_schedule_with_warmup, num_warmup_steps=args.num_warmup_steps
    )
    optimizer_partial_fn = partial(torch.optim.AdamW, lr=args.learning_rate, weight_decay=0.01)  #

    logger.info("Creating the lightning module")
    lightning_module = ClusterLightningModule(
        model=model,
        optimizer_partial_fn=optimizer_partial_fn,
        lr_scheduler_partial_fn=learning_rate_scheduler_partial,
        annotation_covab_size=get_annotation_vocab_size(),
    )

    return lightning_module


def main() -> None:
    """Train the model."""
    seed_everything(SEED)
    args = arguments_parser()
    torch.set_float32_matmul_precision("high")

    datamodule = MTLDataModule(
        num_workers=17,
        datasetname=args.dataset,
        datapath=DATASETPATH,
        batch_size=32,
        dataloader_kwargs={},
    )
    lightning_module = create_model(args)
    model_checkpoint_path = create_checkpoint_path(dataset_name=DatasetName(args.dataset))

    logger.info("Setting up wandb.")
    wandb_logger = WandbLogger(
        project="cluster",
        entity="lewidi",
        log_model=False,
        name=f"{args.model} - {args.dataset}" if SWEEP_ID is None else None,
        group="Best Models" if SWEEP_ID is not None else None,
        config={
            **vars(args),
            "dataset_name": args.dataset,
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
        RichModelSummary(-1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        early_stop_callback,
    ]
    if ENABLE_CHECKPOINTING:
        callbacks.append(checkpoint_callback)

    logger.info("Setting up trainer")
    trainer = pl.Trainer(
        max_epochs=500,
        # accumulate_grad_batches=4,
        callbacks=callbacks,
        gradient_clip_val=1,
        log_every_n_steps=10,
        logger=wandb_logger,
    )

    logger.info("Starting the training...")
    trainer.fit(lightning_module, datamodule=datamodule)
    logger.info("Done.")


if __name__ == "__main__":
    main()
