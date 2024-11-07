import itertools
import json
import operator
from enum import Enum
from pathlib import Path
from typing import Any, Self, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from mtl_cluster.data.datasets.constants import DatasetName


class Gender(Enum):
    """gender of annotator."""

    male = "Male"
    female = "Female"
    other = "Other/Prefer not to say"


class Education(Enum):
    """all possible education levels in the dataset."""

    bachelors = "Bachelors degree"
    vocational = "Vocational or technical school"
    associate_degree = "Associate degree"
    graduate = "Graduate work"
    college_degree = "Some college"
    highschool_graduate = "High school graduate"
    some_highschool = "Some high school"
    undisclosed = "I prefer not to say"


class NativeSpeaker(Enum):
    """Whether the person is a native english speaker or not."""

    native_speaker = "Native speaker"
    near_native = "Near-native speaker"
    non_native = "Non-native speaker"


class BiasLabel(Enum):
    """Coding of agreement / disagreement / neutral stance."""

    non_biased = "Non-biased"
    biased = "Biased"


class FactualLabel(Enum):
    """Factual value of each sentence."""

    factual = "Entirely factual"
    somewhat_factual = "Somewhat factual but also opinionated"
    opinion = "Expresses writers opinion"


class SentenceInstance(BaseModel):
    """Instances of each sentence."""

    model_config = ConfigDict(frozen=True)

    sentence: str
    sentence_id: str
    sent_outlet: str

    mturk_id: str
    age: int
    political_ideology: int

    bias_label: BiasLabel
    factual_label: FactualLabel
    gender: Gender
    education: Education
    native_speaker: NativeSpeaker
    news_outlets: list[str]

    @classmethod
    def from_raw_instance(cls, raw_annotation: dict[str, Any]) -> Self:
        """Create the instance from a raw row in the spreadsheet.

        For some of the inputs, the raw data used smart quotes, which are annoying to work with.
        So they have just been removed.
        """
        return cls(
            sentence=raw_annotation["text"],
            sentence_id=raw_annotation["sentence_id"],
            sent_outlet=raw_annotation["outlet"],
            mturk_id=raw_annotation["mturk_id"],
            age=raw_annotation["age"],
            political_ideology=raw_annotation["political_ideology"],
            bias_label=BiasLabel(raw_annotation["label"]),
            factual_label=FactualLabel(raw_annotation["factual"].replace("\u2019", "")),
            gender=Gender(raw_annotation["gender"]),
            education=Education(raw_annotation["education"].replace("\u2019", "")),
            native_speaker=NativeSpeaker(raw_annotation["native_english_speaker"]),
            news_outlets=raw_annotation["followed_news_outlets"],
        )


class Annotator(BaseModel):
    """Instance of annotator per sentence."""

    mturk_id: str
    age: int
    political_ideology: int
    gender: Gender
    education: Education
    native_speaker: NativeSpeaker
    news_outlets: list[str]

    @classmethod
    def from_sentence_instance(cls, sentence_instance: SentenceInstance) -> Self:
        """Initialize from a single sentence instance."""
        return cls(
            mturk_id=sentence_instance.mturk_id,
            age=sentence_instance.age,
            political_ideology=sentence_instance.political_ideology,
            gender=sentence_instance.gender,
            education=sentence_instance.education,
            native_speaker=sentence_instance.native_speaker,
            news_outlets=sentence_instance.news_outlets,
        )


class Annotation(BaseModel):
    """Instance of an annotation par (bias label and factual label)."""

    bias_label: BiasLabel
    factual_label: FactualLabel

    @classmethod
    def from_sentence_instance(cls, sentence_instance: SentenceInstance) -> Self:
        """Initialise an annotation from a single sentence instance."""
        return cls(
            bias_label=sentence_instance.bias_label, factual_label=sentence_instance.factual_label
        )


class AnonSentenceInstance(BaseModel):
    """Instance of the anonymous dataset to be stored into json."""

    sentence_id: str
    sentence: str
    sent_outlet: str
    annotators: list[Annotator]
    annotations: list[Annotation]
    dataset_name : DatasetName = DatasetName.anon

    @classmethod
    def from_sentence_instances(cls, instances: list[SentenceInstance]) -> Self:
        """Instantiate from multiple sentence instances."""
        annotators = list(map(Annotator.from_sentence_instance, instances))
        annotations = list(map(Annotation.from_sentence_instance, instances))

        return cls(
            sentence_id=instances[0].sentence_id,
            sentence=instances[0].sentence,
            sent_outlet=instances[0].sent_outlet,
            annotators=annotators,
            annotations=annotations,
        )


def group_instances_by_sentence_id(
    all_instances: list[SentenceInstance],
) -> list[list[SentenceInstance]]:
    """Group all the sentence instances by the sentence ID."""
    keyfunc = operator.attrgetter("sentence_id")
    grouped_sentences = {
        k: list(v) for k, v in itertools.groupby(sorted(all_instances, key=keyfunc), keyfunc)
    }
    return list(grouped_sentences.values())


def main() -> None:
    """Reads the related files, processes and dumps them into Json."""
    storage_dir = Path("storage/data/Anonymous/old_csv/")
    output_file = Path("storage/data/modified/into_json/Anon_parsed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotations = storage_dir.joinpath("annotations.csv")

    annotations = pd.read_csv(
        annotations, sep=",", header=0, converters={"followed_news_outlets": pd.eval}
    )

    sentence_instances: list[SentenceInstance] = [
        SentenceInstance.from_raw_instance(cast(dict[str, Any], raw_sentence_instance))
        for raw_sentence_instance in annotations.to_dict("records")
    ]
    grouped_sentence_instances: list[list[SentenceInstance]] = group_instances_by_sentence_id(sentence_instances)

    output_instances: list[AnonSentenceInstance] = [
        AnonSentenceInstance.from_sentence_instances(instances)
        for instances in grouped_sentence_instances
    ]

    instances_as_dicts = [instance.model_dump_json() for instance in output_instances]

    output_file.write_text(json.dumps(instances_as_dicts))


if __name__ == "__main__":
    main()
