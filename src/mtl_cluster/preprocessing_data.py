import json
import pickle
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TypeVar

from mtl_cluster.data.dataset_instance import ModelInstance
from mtl_cluster.data.datasets.anonymous_into_json import AnonSentenceInstance
from mtl_cluster.data.datasets.babe_into_json import BabeSentenceInstance
from mtl_cluster.data.datasets.constants import DatasetName
from mtl_cluster.data.datasets.gwsd_into_json import GwsdSentenceInstance
from mtl_cluster.data.datasets.mbic_into_json import MBICSentenceInstance
from mtl_cluster.data.split import create_dataset_split
from mtl_cluster.modules.tokenise_split.tokenisers import (
    AnnotationTokenizer,
    AnnotatorTokenizer,
    Preprocessor,
    TextTokenizer,
    annotation_label_vocabulary_build,
)


S = TypeVar("S", bound=Enum)
T = TypeVar("T")

GWSD_DATA_PATH = Path("storage/data/modified/into_json/gwsd_parsed.json")
MBIC_DATA_PATH = Path("storage/data/modified/into_json/MBIC_parsed.json")
ANON_DATA_PATH = Path("storage/data/modified/into_json/Anon_parsed.json")
BABE_DATA_PATH = Path("storage/data/modified/into_json/BABE_parsed.json")

MODEL_DATA_PATH = Path("storage/data/model_data/")
TRAIN_PATH = MODEL_DATA_PATH.joinpath("train/")
VAL_PATH = MODEL_DATA_PATH.joinpath("val/")
TEST_PATH = MODEL_DATA_PATH.joinpath("test/")
UNIQUE_ANN_PATH = MODEL_DATA_PATH.joinpath("unique_annotators/")
TOKENIZER_PATH = MODEL_DATA_PATH.joinpath("tokenizers/")

TRAIN_PATH.mkdir(parents=True, exist_ok=True)
VAL_PATH.mkdir(parents=True, exist_ok=True)
TEST_PATH.mkdir(parents=True, exist_ok=True)
UNIQUE_ANN_PATH.mkdir(parents=True, exist_ok=True)
TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)

UNIQUE_ANNOTATION_PATH_ANON = UNIQUE_ANN_PATH.joinpath("anon_annotator_metadata_full.json")
UNIQUE_ANNOTATION_PATH_MBIC = UNIQUE_ANN_PATH.joinpath("mbic_annotator_metadata_full.json")


def create_model_instances_from_gwsd() -> list[ModelInstance]:
    """Create model instances from dataset."""
    # Load the GWSD data from the path for it and create the parsed pydantic instances
    gwsd_raw_instances: list[str] = json.loads(GWSD_DATA_PATH.read_text())
    gwsd_parsed_instances = [
        GwsdSentenceInstance.model_validate_json(instance) for instance in gwsd_raw_instances
    ]
    gwsd_model_instances = [
        ModelInstance.from_gwsd(instances) for instances in gwsd_parsed_instances
    ]
    return gwsd_model_instances


def create_ids_fromstring_dict_anon(instances: list[AnonSentenceInstance]) -> dict[str, int]:
    """Create a dictionary of unique annotators from string ids to int."""
    unique_annotators = {
        annotator.mturk_id for instance in instances for annotator in instance.annotators
    }
    return {annotator: index for index, annotator in enumerate(sorted(unique_annotators))}


def create_ids_fromstring_dict_mbic(instances: list[MBICSentenceInstance]) -> dict[str, int]:
    """Create a dictionary of unique annotators from string ids to int."""
    unique_annotators = {
        annotator.mturk_id for instance in instances for annotator in instance.annotators
    }
    return {annotator: index for index, annotator in enumerate(sorted(unique_annotators))}


def get_annotator_id_vocab_anon(
    *, parsed_instances: list[AnonSentenceInstance], force: bool = False
) -> dict[str, int]:
    if not UNIQUE_ANNOTATION_PATH_ANON.exists() or force:
        id_dict = create_ids_fromstring_dict_anon(parsed_instances)
        UNIQUE_ANNOTATION_PATH_ANON.write_text(json.dumps(id_dict))
        return id_dict
    return json.loads(UNIQUE_ANNOTATION_PATH_ANON.read_text())


def get_annotator_id_vocab_mbic(
    *, parsed_instances: list[MBICSentenceInstance], force: bool = False
) -> dict[str, int]:
    if not UNIQUE_ANNOTATION_PATH_MBIC.exists() or force:
        id_dict = create_ids_fromstring_dict_mbic(parsed_instances)
        UNIQUE_ANNOTATION_PATH_MBIC.write_text(json.dumps(id_dict))
        return id_dict
    return json.loads(UNIQUE_ANNOTATION_PATH_MBIC.read_text())


def create_model_instances_from_anon() -> list[ModelInstance]:
    """Create model instances for the Anon dataset."""
    raw_instances: list[str] = json.loads(ANON_DATA_PATH.read_text())
    parsed_instances = [
        AnonSentenceInstance.model_validate_json(instance) for instance in raw_instances
    ]

    anon_annotator_id_vocab = get_annotator_id_vocab_anon(parsed_instances=parsed_instances)
    model_instances = [
        ModelInstance.from_anon(instance, anon_annotator_id_vocab) for instance in parsed_instances
    ]
    return model_instances


def create_mode_instances_from_mbic() -> list[ModelInstance]:
    """Create model instances for the MBIC dataset."""
    raw_instances: list[str] = json.loads(MBIC_DATA_PATH.read_text())

    parsed_instances = [
        MBICSentenceInstance.model_validate_json(instance) for instance in raw_instances
    ]

    mbic_annotator_id_vocab = get_annotator_id_vocab_mbic(parsed_instances=parsed_instances)
    model_instances = [
        ModelInstance.from_mbic(instance, mbic_annotator_id_vocab) for instance in parsed_instances
    ]
    return model_instances


def create_mode_instances_from_babe() -> list[ModelInstance]:
    """Create model instances for the BABE dataset."""
    raw_instances: list[str] = json.loads(BABE_DATA_PATH.read_text())
    parsed_instances = [
        BabeSentenceInstance.model_validate_json(instance) for instance in raw_instances
    ]

    model_instances = [ModelInstance.from_babe(instance) for instance in parsed_instances]
    return model_instances


def main() -> None:
    """Split datasets + preprocess."""
    dataset_name = DatasetName.babe

    # TODO: Understand this
    switcher: dict[DatasetName, Callable[[], list[ModelInstance]]] = {
        DatasetName.gwsd: create_model_instances_from_gwsd,
        DatasetName.anon: create_model_instances_from_anon,
        DatasetName.mbic: create_mode_instances_from_mbic,
        DatasetName.babe: create_mode_instances_from_babe,
    }
    model_instances = switcher[dataset_name]()

    # Load data from the path for it and create the parsed pydantic instances
    #  of instances differentiated only by DatasetName and iterate through that.
    train_instances, val_instances = create_dataset_split(
        instances=model_instances, train_size=0.9, val_size=0.1
    )

    vocab = annotation_label_vocabulary_build()
    annotation_tokenizer = AnnotationTokenizer(vocab)

    annotator_metadata = {}

    for instance in model_instances:
        for annotator, annotator_info in instance.annotator_info_per_instance.items():
            if annotator not in list(annotator_metadata.keys()):
                annotator_metadata[annotator] = annotator_info

    unique_annotator_id_list = list(annotator_metadata.keys())
    annotator_tokenizer = AnnotatorTokenizer.from_unique_annotators(unique_annotator_id_list)

    text_tokenizer = TextTokenizer()

    # # Instantiate the annotation tokenizer

    # Instantiate the preprocessor
    instance_preprocessor = Preprocessor(
        text_tokenizer=text_tokenizer,
        annotator_tokenizer=annotator_tokenizer,
        annotation_tokenizer=annotation_tokenizer,
    )

    # Tokenize gwsd instances into text/ annotator seprate instances for encoder / decoder model
    tokenised_separate_train_instances = [
        instance_preprocessor.preprocess_separate(instance) for instance in train_instances
    ]

    tokenised_separate_val_instances = [
        instance_preprocessor.preprocess_separate(instance) for instance in val_instances
    ]

    with TRAIN_PATH.joinpath(f"{dataset_name.value}_separate_train.pkl").open("wb") as f:
        pickle.dump(tokenised_separate_train_instances, f)

    # model_instance_adapter = TypeAdapter(list[ModelInstance])
    # TRAIN_PATH.joinpath(f"{dataset_name.value}_train_unencoded.json").write_bytes(
    #     model_instance_adapter.dump_json(train_instances)
    # )

    # VAL_PATH.joinpath(f"{dataset_name.value}_val_unencoded.json").write_bytes(
    #     model_instance_adapter.dump_json(val_instances)
    # )
    with TRAIN_PATH.joinpath(f"{dataset_name.value}_train_unencoded.pkl").open("wb") as f:
        pickle.dump(train_instances, f)

    with VAL_PATH.joinpath(f"{dataset_name.value}_val_unencoded.pkl").open("wb") as f:
        pickle.dump(val_instances, f)

    with VAL_PATH.joinpath(f"{dataset_name.value}_separate_val.pkl").open("wb") as f:
        pickle.dump(tokenised_separate_val_instances, f)

    with UNIQUE_ANN_PATH.joinpath(f"{dataset_name.value}_annotator_metadata.pkl").open("wb") as f:
        pickle.dump(annotator_metadata, f)

    TOKENIZER_PATH.joinpath(f"{dataset_name.value}_annotator_vocab.json").write_text(
        json.dumps(annotator_tokenizer.vocab)
    )


if __name__ == "__main__":
    main()
