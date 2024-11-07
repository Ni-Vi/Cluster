from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange
from torchmetrics import Metric
from torchmetrics.classification import MulticlassExactMatch
from torchmetrics.functional import pairwise_cosine_similarity

from mtl_cluster.data.datamodule import DatasetOutput
from mtl_cluster.models.models import ModelProtocol


OptimizerPartialFn = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
LRSchedulerPartialFn = Callable[..., torch.optim.lr_scheduler.LRScheduler]

_default_optimizer = torch.optim.Adam
_default_lr_scheduler = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)


class ClusterLightningModule(pl.LightningModule):
    """Lighting module for training the cluster model offline."""

    def __init__(
        self,
        model: ModelProtocol,
        *,
        annotation_covab_size: int,
        optimizer_partial_fn: OptimizerPartialFn = _default_optimizer,
        lr_scheduler_partial_fn: LRSchedulerPartialFn = _default_lr_scheduler,
    ) -> None:
        super().__init__()

        self._accuracy = MulticlassExactMatch(num_classes=annotation_covab_size, ignore_index=-100)
        self._optimizer_partial_fn = optimizer_partial_fn
        self._lr_scheduler_partial_fn = lr_scheduler_partial_fn
        self.model = model
        try:
            self.model_is_pooling = bool(
                self.model.pooling  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            self.model_is_pooling = False
        self.pairwise_embedding_similarity = PairwiseSimilarity()
        self.save_hyperparameters(logger=False)

    def forward(self, batch: DatasetOutput) -> torch.Tensor:
        """Predict the labels for the current input.

        the forward method just predicts the labels and does nothing else.
        """
        return self.model.forward(
            text_tokens=batch.text,
            text_mask=batch.text_mask,
            annotator_tokens=batch.annotators,
            annotator_mask=batch.annotators_mask,
        )

    def training_step(self, batch: DatasetOutput, batch_idx: int) -> torch.Tensor:
        """Training step."""
        predicted_labels = self.forward(batch)
        assert batch.annotations is not None

        if self.model_is_pooling:
            # we need to extract the mode of every row to get the most common annotation, and as
            # such must handle our -100 masks
            pooled_annotations_mask = torch.randint_like(batch.annotations, low=10, high=10000)
            annotations_post_mask = torch.where(
                batch.annotations == -100, pooled_annotations_mask, batch.annotations
            )
            modes, _ = torch.mode(annotations_post_mask, dim=1)
            loss = torch.nn.functional.cross_entropy(
                predicted_labels.view(-1, predicted_labels.size(-1)), modes.view(-1)
            )
            accuracy = self._accuracy(
                predicted_labels.view(-1, predicted_labels.size(-1)), modes.view(-1)
            )

        else:
            loss = torch.nn.functional.cross_entropy(
                predicted_labels.view(-1, predicted_labels.size(-1)), batch.annotations.view(-1)
            )
            # pred labels  = (batch_size , num annotators , num_classes)

            accuracy = self._accuracy(
                predicted_labels.view(-1, predicted_labels.size(-1)), batch.annotations.view(-1)
            )
        self.log("train_acc", accuracy, prog_bar=True, logger=True, batch_size=batch.text.size(0))
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=batch.text.size(0))
        return loss

    def validation_step(self, batch: DatasetOutput, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        predicted_labels = self.forward(batch)
        assert batch.annotations is not None
        if self.model_is_pooling:
            # we need to extract the mode of every row to get the most common annotation, and as
            # such must handle our -100 masks
            pooled_annotations_mask = torch.randint_like(batch.annotations, low=10, high=10000)
            annotations_post_mask = torch.where(
                batch.annotations == -100, pooled_annotations_mask, batch.annotations
            )
            modes, _ = torch.mode(annotations_post_mask, dim=1)
            loss = torch.nn.functional.cross_entropy(
                predicted_labels.view(-1, predicted_labels.size(-1)), modes.view(-1)
            )
            accuracy = self._accuracy(
                predicted_labels.view(-1, predicted_labels.size(-1)), modes.view(-1)
            )

        else:
            loss = torch.nn.functional.cross_entropy(
                predicted_labels.view(-1, predicted_labels.size(-1)), batch.annotations.view(-1)
            )
            # pred labels  = (batch_size , num annotators , num_classes)

            accuracy = self._accuracy(
                predicted_labels.view(-1, predicted_labels.size(-1)), batch.annotations.view(-1)
            )
        # self._mutliclass_confusion_matrix.update(
        #     predicted_labels.reshape(-1, predicted_labels.size(-1)),
        #     batch.annotations.reshape(-1, batch.annotations.size(-1)),
        # )
        embedding_space = self.predict_step(batch, batch_idx=0, dataloader_idx=0)
        self.pairwise_embedding_similarity.update(embedding_space, batch.annotators_mask)
        self.log("val_acc", accuracy, prog_bar=True, logger=True, batch_size=batch.text.size(0))
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch.text.size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        self.log_dict(self.pairwise_embedding_similarity.compute(), logger=True)
        self.pairwise_embedding_similarity.reset()

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        optimizer = self._optimizer_partial_fn(self.parameters())
        scheduler = self._lr_scheduler_partial_fn(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def predict_step(
        self, batch: DatasetOutput, batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
        """Predict step, extracts the latent state for every annotator."""
        encoded_annotator_ids = self.model.decode_annotator_ids(
            text_tokens=batch.text,
            text_mask=batch.text_mask,
            annotator_tokens=batch.annotators,
            annotator_mask=batch.annotators_mask,
        )
        # Shape (batch size, max num annotators, embedding size)
        return encoded_annotator_ids

    def _pool_targets(self, targets: torch.Tensor, targets_masked: torch.Tensor) -> torch.Tensor:
        """Get the aggregated target for each batch."""
        pooled_targets = targets.masked_fill(targets_masked, 0)

        _, indices = torch.unique(pooled_targets, dim=1, return_inverse=True)
        mode_indices = torch.argmax(
            torch.bincount(indices, minlength=targets_masked.size(1)), dim=-1
        )
        mode_values = targets_masked[torch.arange(targets_masked.size(0)), mode_indices]
        return mode_values


class PairwiseSimilarity(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("embedding_space", default=[], dist_reduce_fx="cat")
        self.add_state("embedding_mask", default=[], dist_reduce_fx="cat")

    def update(self, embedding_space: torch.Tensor, embedding_mask: torch.Tensor | None) -> None:
        """Update the metric state.

        We are enforcing the shape of the embedding space to be (batch size, num annotators, embedding size).
        """
        if embedding_mask is None:
            embedding_mask = torch.zeros_like(embedding_space[..., 0]).bool()

        assert embedding_space.ndim == 3
        assert embedding_mask.ndim == 2

        embedding_space = rearrange(
            embedding_space, "batch num_annotators dim -> (batch num_annotators) dim"
        )
        embedding_mask = rearrange(
            embedding_mask, "batch num_annotators -> (batch num_annotators)"
        )

        self.embedding_space.append(embedding_space)
        self.embedding_mask.append(embedding_mask)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute pairwise distance metrics."""
        # Shape (batch size X max num annotators, embedding size)
        merged_embedding_space = torch.cat(self.embedding_space, dim=0)
        # Shape (batch size X  max num annotators)
        merged_mask = torch.cat(self.embedding_mask, dim=0)

        # Shape (num annotators that annotated texts, embedding size)
        latent_spaces = merged_embedding_space[~merged_mask]

        pairwise_similarity_matrix = pairwise_cosine_similarity(latent_spaces)
        mask = torch.ones_like(pairwise_similarity_matrix)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.bool()
        pairwise_similarity_matrix = pairwise_similarity_matrix.masked_select(mask)
        similarity_metrics = {
            "val/pairwise_similarity/mean": pairwise_similarity_matrix.mean(),
            "val/pairwise_similarity/std": pairwise_similarity_matrix.std(),
            "val/pairwise_similarity/max": pairwise_similarity_matrix.max(),
            "val/pairwise_similarity/min": pairwise_similarity_matrix.min(),
            "val/pairwise_similarity/median": pairwise_similarity_matrix.median(),
            "val/pairwise_similarity/lower_quartile_range": pairwise_similarity_matrix.kthvalue(  # noqa: PD011
                int(len(pairwise_similarity_matrix) * 0.25)
            ).values,
            "val/pairwise_similarity/upper_quartile_range": pairwise_similarity_matrix.kthvalue(  # noqa: PD011
                int(len(pairwise_similarity_matrix) * 0.75)
            ).values,
        }

        return similarity_metrics
