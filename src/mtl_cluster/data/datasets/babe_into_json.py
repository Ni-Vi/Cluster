import itertools
import json
import operator
import random
import string
from enum import Enum
from pathlib import Path
from typing import Any, Self, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

from mtl_cluster.data.datasets.constants import DatasetName


class Gender(Enum):
    """gender of annotator."""

    male = "male"
    female = "female"


class Education(Enum):
    """all possible education levels in the dataset."""

    bachelors = "Bachelor's degree"


class NativeSpeaker(Enum):
    """Whether the person is a native english speaker or not."""

    native_speaker = "Native speaker"
    near_native = "Near-native speaker"
    non_native_speaker = "Non-native speaker"
    non_native = "Non-native"


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
    daily_stormer = "daily-stormer"
    new_york_times = "new-york-times"
    daily_beast = "daily-beast"

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
            Outlet.daily_stormer: OutletBias.right,
            Outlet.new_york_times: OutletBias.left,
            Outlet.daily_beast: OutletBias.left,
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
    marriage_equality = "marriage-equality"
    blm = "black-lives-matter"
    universal_healthcare = "universal-health-care"
    islam = "islam"
    taxes = "taxes"
    metoo = "#metoo"


class FactualLabel(Enum):
    """Coding of factual label of the sentence as gauged by the annotator."""

    entirely_factual = "Entirely factual"
    no_agreement = "No agreement"
    expresses_writers_opinion = "Expresses writers opinion"
    somewhat_factual_but_also_opinionated = "Somewhat factual but also opinionated"


def _compute_sentence_id(sentence: str) -> str:
    """Compute the sentence ID from the sentence."""
    return "".join(
        c
        for c in list(sentence.strip().lower().replace("\xa0", " "))
        if c in string.ascii_lowercase
    )


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
    annotator_id: int

    # Annotation related
    bias_label: BiasLabel
    factual_label: FactualLabel

    @classmethod
    def parse_raw_instance(cls, raw_annotation: dict[str, Any]) -> Self:
        """Create the instance from a raw row in the spreadsheet.

        For some of the inputs, the raw data used smart quotes, which are annoying to work with.
        So they have just been removed.
        """
        # assert raw_annotation["label_bias"] == nan

        if raw_annotation["topic"] == "vaccine":
            raw_annotation["topic"] = "vaccines"
        if raw_annotation["topic"] == "blm":
            raw_annotation["topic"] = "black-lives-matter"

        if raw_annotation["label_bias"] == "Biased, unfair":
            raw_annotation["label_bias"] = "Biased"
        bias_label = BiasLabel(raw_annotation["label_bias"])

        outlet = Outlet(raw_annotation["outlet"].replace(" ", "-").lower())

        try:
            factual_label = FactualLabel(raw_annotation["label_opinion"])
        except ValueError as err:
            if raw_annotation["label_opinion"] in {
                "Expresses wleter´s opinion",
                "Expresses writer’s opinion",
            }:
                factual_label = FactualLabel.expresses_writers_opinion
            elif raw_annotation["label_opinion"] == "Somewhat factional but also opinionated":
                factual_label = FactualLabel.somewhat_factual_but_also_opinionated
            else:
                raise ValueError("Unhanded factual label.") from err

        return cls(
            sentence=raw_annotation["text"].strip().replace("\xa0", " "),
            sentence_id=_compute_sentence_id(raw_annotation["text"]),
            sent_outlet=outlet,
            sent_topic=Topic(raw_annotation["topic"].replace(" ", "-").lower()),
            sent_outlet_bias=outlet.bias,
            annotator_id=raw_annotation["annotator_id"],
            bias_label=bias_label,
            factual_label=factual_label,
        )


class Annotator(BaseModel):
    """Instance of annotators as per the annotator demographics file."""

    model_config = ConfigDict(frozen=True)

    annotator_id: int
    political_ideology: int
    gender: Gender
    education: Education
    native_speaker: NativeSpeaker
    news_outlets: str


def parse_raw_annotator(raw_annotation: dict[str, Any]) -> Annotator:
    """Initialize from a single sentence instance."""
    return Annotator(
        annotator_id=raw_annotation["annotator_id"],
        political_ideology=raw_annotation["political orientation"],
        gender=Gender(raw_annotation["gender"]),
        education=Education(raw_annotation["education"]),
        native_speaker=NativeSpeaker(raw_annotation["English proficiency"]),
        news_outlets=raw_annotation["news outlets"],
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


def group_instances_by_sentence_text(
    all_instances: list[SentenceInstance],
) -> list[list[SentenceInstance]]:
    """Group all the sentence instances by the sentence ID."""
    keyfunc = operator.attrgetter("sentence_id")
    keyfunc2 = operator.attrgetter("annotator_id")

    grouped_sentences = {}

    for sentence_id, instances_iter in itertools.groupby(
        sorted(all_instances, key=keyfunc), keyfunc
    ):
        instances_for_sentence = []

        for _, annotator_instances_iter in itertools.groupby(
            sorted(instances_iter, key=keyfunc2), keyfunc2
        ):
            instances_for_annotator_id = list(annotator_instances_iter)
            if len(instances_for_annotator_id) > 1:
                people_are_identical = all(
                    instance.annotator_id == instances_for_annotator_id[0].annotator_id
                    and instance.bias_label == instances_for_annotator_id[0].bias_label
                    and instance.factual_label == instances_for_annotator_id[0].factual_label
                    for instance in instances_for_annotator_id
                )
                if not people_are_identical:
                    instances_for_annotator_id = [random.choice(instances_for_annotator_id)]  # noqa: S311
                else:
                    instances_for_annotator_id = [instances_for_annotator_id[0]]

            instances_for_sentence.extend(instances_for_annotator_id)

        grouped_sentences[sentence_id] = instances_for_sentence

    return list(grouped_sentences.values())


class BabeSentenceInstance(BaseModel):
    """Instances of each sentence."""

    sentence: str
    sentence_id: str
    sent_outlet: Outlet
    sent_topic: Topic
    sent_outlet_bias: OutletBias
    annotators: list[Annotator]
    annotations: list[Annotation]
    dataset_name: DatasetName

    @classmethod
    def from_sentence_instances(
        cls, instances: list[SentenceInstance], all_annotators: list[Annotator]
    ) -> Self:
        """Instantiate from multiple sentence instances."""
        annotations = list(map(Annotation.from_sentence_instance, instances))
        sentence_annotator_ids = [instance.annotator_id for instance in instances]

        annotators = [
            annotator
            for annotator_ids in sentence_annotator_ids
            for annotator in all_annotators
            if annotator.annotator_id == annotator_ids
        ]

        return cls(
            dataset_name=DatasetName.babe,
            sentence=instances[0].sentence,
            sentence_id=instances[0].sentence_id,
            sent_outlet=instances[0].sent_outlet,
            sent_topic=instances[0].sent_topic,
            sent_outlet_bias=instances[0].sent_outlet_bias,
            annotators=annotators,
            annotations=annotations,
        )


def to_babe_sentence_instances(
    instances: list[SentenceInstance], annotators: list[Annotator]
) -> BabeSentenceInstance:
    """Return the instance of the babe dataset."""
    sentence_id = instances[0].sentence_id
    raise NotImplementedError


def main() -> None:
    """Reads the related files, processes and dumps them into Json."""
    storage_dir = Path("storage/data/BABE/old_csv/")
    output_file = Path("storage/data/modified/into_json/BABE_parsed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotations_sg1 = storage_dir.joinpath("raw_labels_SG1.csv")
    annotations_sg2 = storage_dir.joinpath("raw_labels_SG2.csv")

    annotators = storage_dir.joinpath("annotator_demographics.csv")

    annotations_sg1 = pd.read_csv(annotations_sg1, sep=";", header=0)
    annotations_sg2 = pd.read_csv(annotations_sg2, sep=";", header=0)

    annotators = pd.read_csv(annotators, sep=";", header=0)
    annotators_as_list = annotators.to_dict("records")

    parsed_annotators = list(map(parse_raw_annotator, annotators_as_list))

    babe_sentences = pd.concat([annotations_sg1, annotations_sg2])

    babe_sentences = babe_sentences.dropna(subset=["label_bias", "label_opinion", "topic"])

    sentence_instances = []
    for raw_sentence_instance in babe_sentences.to_dict("records"):
        sentence_instances.append(
            SentenceInstance.parse_raw_instance(cast(dict[str, Any], raw_sentence_instance))
        )

    grouped_sentence_instances: list[list[SentenceInstance]] = group_instances_by_sentence_text(
        sentence_instances
    )

    output_instances = [
        BabeSentenceInstance.from_sentence_instances(instances, all_annotators=parsed_annotators)
        for instances in grouped_sentence_instances
    ]

    instances_as_dicts = [instance.model_dump_json() for instance in output_instances]

    output_file.write_text(json.dumps(instances_as_dicts))


if __name__ == "__main__":
    main()
