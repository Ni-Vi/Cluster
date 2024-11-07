import itertools
import json
import operator
from enum import Enum
from pathlib import Path
from typing import Any, Self

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


class OutletBias(Enum):
    """Coding of political stance of the outlet and thus sentence."""

    left = "left"
    lean_left = "lean-left"
    center = "center"
    lean_right = "lean-right"
    right = "right"


class Outlet(Enum):
    """Coding of agreement / disagreement / neutral stance."""

    msnbc = "msnbc"
    usa_today = "usa-today"
    breitbart = "breitbart"
    federalist = "federalist"
    huffpost = "huffpost"
    fox_news = "fox-news"
    reuters = "reuters"
    alternet = "alternet"

    @property
    def bias(self) -> OutletBias:
        """Get the bias for the current outlet."""
        switcher = {
            Outlet.msnbc: OutletBias.left,
            Outlet.usa_today: OutletBias.lean_left,
            Outlet.breitbart: OutletBias.right,
            Outlet.federalist: OutletBias.right,
            Outlet.huffpost: OutletBias.left,
            Outlet.fox_news: OutletBias.right,
            Outlet.reuters: OutletBias.center,
            Outlet.alternet: OutletBias.left,
        }

        return switcher[self]


class Topic(Enum):
    """Coding of topic of the sentence."""

    environment = "environment"
    abortion = "abortion"
    immigration = "immigration"
    international_politics_and_world_news = "international-politics-and-world-news"
    gun_control = "gun-control"
    white_nationalism = "white-nationalism"
    vaccines = "vaccines"
    sport = "sport"
    middle_class = "middle-class"
    gender = "gender"
    elections_2020 = "elections-2020"
    trump_presidency = "trump-presidency"
    student_debt = "student-debt"
    coronavirus = "coronavirus"


class FactualLabel(Enum):
    """Coding of factual label of the sentence as gauged by the annotator."""

    entirely_factual = "Entirely factual"
    no_agreement = "No agreement"
    expresses_writers_opinion = "Expresses writers opinion"
    somewhat_factual_but_also_opinionated = "Somewhat factual but also opinionated"


class SentenceInstance(BaseModel):
    """Single instance of the MBIC Labeled Dataset."""

    model_config = ConfigDict(frozen=True)
    # Sentence related
    sentence: str
    sentence_id: str
    sent_outlet: Outlet
    sent_topic: Topic
    sent_outlet_bias: OutletBias

    # Annotator related
    mturk_id: str
    age: int
    political_ideology: int
    gender: Gender
    education: Education
    news_outlets: list[str]
    native_speaker: NativeSpeaker

    # Annotation related
    bias_label: BiasLabel
    factual_label: FactualLabel

    @classmethod
    def from_raw_instance(cls, raw_annotation: dict[str, Any]) -> Self:
        """Create the instance from a raw row in the spreadsheet.

        For some of the inputs, the raw data used smart quotes, which are annoying to work with.
        So they have just been removed.
        """
        outlet = Outlet(raw_annotation["outlet"])
        return cls(
            sentence=raw_annotation["text"],
            sentence_id=raw_annotation["sentence_id"],
            sent_outlet=outlet,
            sent_topic=Topic(raw_annotation["topic"]),
            sent_outlet_bias=outlet.bias,
            # article can be found in other file
            mturk_id=raw_annotation["mturk_id"],
            age=raw_annotation["age"],
            political_ideology=raw_annotation["political_ideology"],
            gender=Gender(raw_annotation["gender"]),
            education=Education(raw_annotation["education"].replace("\u2019", "")),
            native_speaker=NativeSpeaker(raw_annotation["native_english_speaker"]),
            news_outlets=raw_annotation["followed_news_outlets"],
            bias_label=BiasLabel(raw_annotation["label_bias"]),
            factual_label=FactualLabel(raw_annotation["label_opinion"].replace("\u2019", "")),
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


class SentenceInfo(BaseModel):
    """Instance of the sentence info."""

    outlet: Outlet
    topic: Topic
    outlet_bias: OutletBias

    @classmethod
    def from_sentence_instance(cls, sentence_instance: SentenceInstance) -> Self:
        """Initialise an annotation from a single sentence instance."""
        return cls(
            outlet=sentence_instance.sent_outlet,
            topic=sentence_instance.sent_topic,
            outlet_bias=sentence_instance.sent_outlet_bias,
        )


class MBICSentenceInstance(BaseModel):
    """Single instance of the Anonymous MBIC dataset."""

    sentence_id: str
    sentence: str
    sentence_info: SentenceInfo
    annotators: list[Annotator]
    annotations: list[Annotation]
    dataset_name: DatasetName = DatasetName.anon

    @classmethod
    def from_sentence_instances(cls, instances: list[SentenceInstance]) -> Self:
        """Instantiate from multiple sentence instances."""
        annotators = list(map(Annotator.from_sentence_instance, instances))
        annotations = list(map(Annotation.from_sentence_instance, instances))
        sent_info = SentenceInfo.from_sentence_instance(instances[0])

        return cls(
            sentence_id=instances[0].sentence_id,
            sentence=instances[0].sentence,
            sentence_info=sent_info,
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
    storage_dir = Path("storage/data/BABE/old_csv/")
    output_file = Path("storage/data/modified/into_json/MBIC_parsed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotations = storage_dir.joinpath("raw_labels_MBIC.csv")

    annotations = pd.read_csv(
        annotations, sep=";", header=0, converters={"followed_news_outlets": pd.eval}
    )

    annotations = annotations.dropna(subset=["label_bias"])
    sentence_instances = [
        SentenceInstance.from_raw_instance(raw_sentence_instance)
        for raw_sentence_instance in annotations.to_dict("records")
    ]
    grouped_sentence_instances = group_instances_by_sentence_id(sentence_instances)

    output_instances = [
        MBICSentenceInstance.from_sentence_instances(instances)
        for instances in grouped_sentence_instances
    ]

    instances_as_dicts = [instance.model_dump_json() for instance in output_instances]

    output_file.write_text(json.dumps(instances_as_dicts))


if __name__ == "__main__":
    main()
