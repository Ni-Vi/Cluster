from enum import Enum
from pathlib import Path
from typing import Generic, Self, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from mtl_cluster.data.datasets.anonymous_into_json import (
    AnonSentenceInstance,
)
from mtl_cluster.data.datasets.babe_into_json import BabeSentenceInstance
from mtl_cluster.data.datasets.constants import DatasetName
from mtl_cluster.data.datasets.gwsd_into_json import GwsdSentenceInstance
from mtl_cluster.data.datasets.mbic_into_json import MBICSentenceInstance


class PreprocessSettings(BaseSettings):
    """Preprocessing related settings, mainly setting up of paths."""

    storage_dir: Path = Path("storage/data/modified/")
    gwsd: Path = storage_dir.joinpath("into_json/GW_parsed.json")
    anon: Path = storage_dir.joinpath("into_json/Anon_parsed.json")

    outputdir: Path = storage_dir.joinpath("tokenised/")


settings = PreprocessSettings()
S = TypeVar("S", bound=Enum)
T = TypeVar("T")


class ModelInstance(BaseModel, Generic[S, T]):
    """Instance of each dataset into a unified format."""

    sentence: str
    annotations_per_annotator: dict[str, S]
    annotator_info_per_instance: dict[str, T]
    sentence_info: dict[str, str]
    dataset_name: DatasetName

    @property
    def annotators(self) -> list[str]:
        return list(self.annotations_per_annotator.keys())

    @property
    def annotations(self) -> list[S]:
        return list(self.annotations_per_annotator.values())

    @property
    def annotator_set_hash(self) -> int:
        return hash(frozenset(self.annotators))

    @classmethod
    def from_anon(cls, instance: AnonSentenceInstance, id_dict: dict[str, int]) -> Self:
        """Instantiate sentence into instances."""
        for annotator in instance.annotators:
            annotator.mturk_id = str(id_dict[annotator.mturk_id])
        return cls(
            sentence=instance.sentence,
            # need to change the next line as there are 2 labels
            annotations_per_annotator={
                annotator.mturk_id: annotation.bias_label
                for annotator, annotation in zip(
                    instance.annotators, instance.annotations, strict=True
                )
            },
            annotator_info_per_instance={
                annotator.mturk_id: annotator for annotator in (instance.annotators)
            },
            sentence_info={instance.sentence: instance.sent_outlet},
            dataset_name=DatasetName.anon,
        )

    @classmethod
    def from_gwsd(cls, instance: GwsdSentenceInstance) -> Self:
        """Instantiate sentence into instances."""
        return cls(
            sentence=instance.sentence,
            # need to change the next line as there are 2 labels
            annotations_per_annotator={
                str(annotators.index): annotations
                for annotators, annotations in zip(
                    instance.annotators, instance.annotations, strict=True
                )
            },
            annotator_info_per_instance={
                str(annotator.index): annotator for annotator in (instance.annotators)
            },
            sentence_info={instance.sentence: ""},
            dataset_name=DatasetName.gwsd,
        )

    @classmethod
    def from_mbic(cls, instance: MBICSentenceInstance, id_dict: dict[str, int]) -> Self:
        """Instantiate MBIC dataset sentence into model instances"""
        for annotator in instance.annotators:
            annotator.mturk_id = str(id_dict[annotator.mturk_id])
        return cls(
            sentence=instance.sentence,
            # need to change the next line as there are 2 labels
            annotations_per_annotator={
                annotator.mturk_id: annotation.bias_label
                for annotator, annotation in zip(
                    instance.annotators, instance.annotations, strict=True
                )
            },
            annotator_info_per_instance={
                annotator.mturk_id: annotator for annotator in (instance.annotators)
            },
            sentence_info={instance.sentence: ""},
            dataset_name=DatasetName.mbic,
        )

    @classmethod
    def from_babe(cls, instance: BabeSentenceInstance) -> Self:
        """Instantiate MBIC dataset sentence into model instances"""
        return cls(
            sentence=instance.sentence,
            # need to change the next line as there are 2 labels
            annotations_per_annotator={
                str(annotator.annotator_id): annotation.bias_label
                for annotator, annotation in zip(
                    instance.annotators, instance.annotations, strict=True
                )
            },
            annotator_info_per_instance={
                str(annotator.annotator_id): annotator for annotator in (instance.annotators)
            },
            sentence_info={instance.sentence: ""},
            dataset_name=DatasetName.babe,
        )
