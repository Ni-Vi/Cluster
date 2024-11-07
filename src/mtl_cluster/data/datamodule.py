import pickle
from pathlib import Path
from typing import Any, NamedTuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from mtl_cluster.modules.tokenise_split.tokenisers import PreprocessedInstance


class DatasetOutput(NamedTuple):
    """Output of the dataset."""

    text: torch.Tensor
    text_mask: torch.Tensor
    annotators: torch.Tensor
    annotators_mask: torch.Tensor
    annotations: torch.Tensor | None
    annotations_mask: torch.Tensor | None


def collate(batch: list[PreprocessedInstance]) -> DatasetOutput:
    """Collate the batch."""
    text = torch.nn.utils.rnn.pad_sequence(
        [instance.text.squeeze() for instance in batch], batch_first=True
    )
    # masking will be done with TRUE == MASKED
    text_mask = (text == 0).to(torch.bool)

    annotators = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(instance.annotators, dtype=torch.long) for instance in batch],
        padding_value=0,
        batch_first=True,
    )
    annotators_mask = (annotators == 0).to(torch.bool)
    assert len(annotators) == len(batch)

    annotations = None
    annotations_mask = None
    # If the first instance has annotations, then all instances have annotations
    if batch[0].annotations is not None:
        annotations = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(instance.annotations, dtype=torch.long) for instance in batch],
            padding_value=-100,
            batch_first=True,
        )

        annotations_mask = (annotations == -100).to(torch.bool)
        # Make sure that the number of annotations is the same as the number of instances otherwise
        # something specific has definitely gone wrong.
        assert len(annotations) == len(batch)

    return DatasetOutput(
        text=text,
        text_mask=text_mask,
        annotators=annotators,
        annotators_mask=annotators_mask,
        annotations=annotations,
        annotations_mask=annotations_mask,
    )


class MTLDataset(Dataset[PreprocessedInstance]):
    """Dataset for the MTL clustering model."""

    def __init__(self, instances: list[PreprocessedInstance]) -> None:
        self._instances = instances

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, index: int) -> PreprocessedInstance:
        return self._instances[index]


class MTLDataModule(LightningDataModule):
    """Datamodule for the MTL clustering model."""

    def __init__(
        self,
        *,
        num_workers: int,
        datasetname: str,
        datapath: Path,
        batch_size: int,
        dataloader_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.datasetname = datasetname
        self._num_workers = num_workers
        self.batch_size = batch_size
        self._dataloader_kwargs = dataloader_kwargs

        self.train_dataset: MTLDataset
        self.valid_dataset: MTLDataset

    # def prepare_data(self) -> None:

    def setup(self, stage: str) -> None:
        """Load the train, val and test datasets."""
        if stage == "fit":
            train_path = self.datapath.joinpath(
                "train/" + self.datasetname + "_separate_train.pkl"
            )
            train_instances = pickle.load(train_path.open("rb"))  # noqa: S301
            self.train_dataset = MTLDataset(
                list(map(PreprocessedInstance.model_validate, train_instances))
            )

            val_path = self.datapath.joinpath("val/" + self.datasetname + "_separate_val.pkl")
            valid_instances = pickle.load(val_path.open("rb"))  # noqa: S301
            self.valid_dataset = MTLDataset(
                list(map(PreprocessedInstance.model_validate, valid_instances))
            )

    def train_dataloader(self) -> DataLoader[DatasetOutput]:
        """Return the train dataloader.

        Dataloader is receiving PreprocessInstance type objects but returning DatasetOutput type.
        The switch is happening inside the collate function, so we're ignoring the seemingly
        clashing of types.
        """
        return DataLoader[DatasetOutput](
            self.train_dataset,  # pyright: ignore[reportGeneralTypeIssues]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=collate,
            **self._dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[DatasetOutput]:
        """Return the validation dataloader."""
        return DataLoader(
            self.valid_dataset,  # pyright: ignore[reportGeneralTypeIssues]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=collate,
            **self._dataloader_kwargs,
        )

    def transfer_batch_to_device(
        self, batch: DatasetOutput, device: torch.device, dataloader_idx: int
    ) -> DatasetOutput:
        """Transfer the batch to the device."""
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return DatasetOutput(*batch)
