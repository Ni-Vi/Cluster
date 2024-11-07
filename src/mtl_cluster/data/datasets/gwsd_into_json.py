import json
from collections.abc import Hashable
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from mtl_cluster.data.datasets.constants import DatasetName


class SentenceId(BaseModel):
    """Sentence ID for the Sentence."""

    round_index: int
    batch_index: int
    sentence_id: str

    def __str__(self) -> str:
        """I hate docstrings."""
        return f"{self.round_index}_{self.batch_index}_{self.sentence_id}"


class Gender(Enum):
    """All possible genders in the dataset."""

    male = "male"
    female = "female"
    abstain = "abstain"


class Education(Enum):
    """All possible education levels in the dataset."""

    college_graduate = "graduated_college"
    high_school_graduate = "graduated_high_school"
    higher_degree = "higher_degree"
    abstain = "abstain"
    other = "other"


class Party(Enum):
    """All possible party affiliations in the dataset."""

    democrat = "democrat"
    other = "other"
    independent = "independent"
    republican = "republican"
    abstain = "abstain"


class Annotator(BaseModel):
    """All annotator related information."""

    model_config = ConfigDict(frozen=True)

    index: int
    party: Party
    batch_index: int
    round_index: int
    age: int
    worker_index: int

    gender: Gender
    education: Education


class AnnotationValue(Enum):
    """Coding of agreement / disagreement / neutral stance."""

    disagrees = -1
    neutral = 0
    agrees = 1


class GwsdSentenceInstance(BaseModel):
    """Instances of each sentence."""

    sentence_id: SentenceId
    sentence: str

    annotators: list[Annotator]
    annotations: list[AnnotationValue]
    dataset_name: DatasetName = DatasetName.gwsd


def parse_raw_annotator(raw_annotator: dict[Hashable, Any]) -> Annotator:
    """Return annotators after parsing."""
    return Annotator(
        index=raw_annotator["index"],
        party=Party(raw_annotator["party"]),
        batch_index=raw_annotator["batch"],
        round_index=raw_annotator["round"],
        gender=Gender(raw_annotator["gender"]),
        age=raw_annotator["age"],
        education=Education(raw_annotator["education"]),
        # Get the worker int from the following string: `round1_batch0_worker0`
        worker_index=int(raw_annotator["unique_id"].split("_worker")[-1]),
    )


def get_annotator_for_sentence(
    all_annotators: list[Annotator], batch_index: int, round_index: int, worker_index: int
) -> Annotator:
    """Get the annotator related to each sentence by matching the round, batch and worker index."""
    for annotator in all_annotators:
        is_correct_annotator = (
            annotator.batch_index == batch_index
            and annotator.round_index == round_index
            and annotator.worker_index == worker_index
        )

        if is_correct_annotator:
            return annotator

    raise AssertionError("Annotator not found")


def parse_raw_sentence_instance(
    raw_sentence_instance: dict[Hashable, Any], *, all_annotators: list[Annotator]
) -> GwsdSentenceInstance:
    """Takes all sentences and pairs them with annotators."""
    sentence_id: str = raw_sentence_instance["sent_id"]
    sentence: str = raw_sentence_instance["sentence"]
    batch_index = raw_sentence_instance["batch"]
    round_index = raw_sentence_instance["round"]

    annotators = []
    annotations = []

    # we know there will always be 8 workers
    for worker_index in range(8):
        annotator = get_annotator_for_sentence(
            all_annotators,
            batch_index=batch_index,
            round_index=round_index,
            worker_index=worker_index,
        )
        raw_annotation = raw_sentence_instance[f"worker_{worker_index}"]
        annotation = AnnotationValue[raw_annotation]

        annotators.append(annotator)
        annotations.append(annotation)

    instance = GwsdSentenceInstance(
        sentence_id=SentenceId(
            batch_index=batch_index, round_index=round_index, sentence_id=sentence_id
        ),
        sentence=sentence,
        annotators=annotators,
        annotations=annotations,
    )
    return instance


def main() -> None:
    """Reads the related files, processes and dumps them into Json."""
    storage_dir = Path("storage/data/GWSTANCE/old_csv/")
    output_file = Path("storage/data/modified/into_json/gwsd_parsed.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    worker_attributes_path = storage_dir.joinpath("3_stance_detection/data/worker_attributes.tsv")
    subjective_info_path = storage_dir.joinpath("1_MTurk/anon_subj_info.tsv")

    worker_attributes = pd.read_csv(worker_attributes_path, sep="\t", header=0)
    subjective_info = pd.read_csv(subjective_info_path, sep="\t", header=0)
    subjective_info = subjective_info.reset_index()

    annotator_full = pd.concat([worker_attributes, subjective_info], axis=1)
    annotator_as_list = annotator_full.to_dict("records")

    parsed_annotators = list(map(parse_raw_annotator, annotator_as_list))

    # full_annotations_path = storage_dir.joinpath("1_MTurk/full_annotations.tsv")
    # full_annotations = pd.read_csv(full_annotations_path, sep="\t", header= 0)

    gwsd_path = storage_dir.joinpath("GWSD.tsv")
    gwsd_dataset = pd.read_csv(gwsd_path, sep="\t", header=0)

    instances = [
        parse_raw_sentence_instance(raw_sentence_instance, all_annotators=parsed_annotators)
        for raw_sentence_instance in gwsd_dataset.to_dict("records")
    ]
    instances_as_dicts = [instance.model_dump_json() for instance in instances]

    output_file.write_text(json.dumps(instances_as_dicts))


if __name__ == "__main__":
    main()
