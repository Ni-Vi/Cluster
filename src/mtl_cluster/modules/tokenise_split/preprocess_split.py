import json
import pickle
from pathlib import Path

from pydantic import TypeAdapter

from mtl_cluster.data.dataset_instance import ModelInstance
from mtl_cluster.data.datasets.constants import DatasetName, Metadata
from mtl_cluster.data.datasets.gwsd_into_json import GwsdSentenceInstance
from mtl_cluster.data.split import create_dataset_split
from mtl_cluster.modules.tokenise_split.tokenisers import (
    AnnotationTokenizer,
    AnnotatorTokenizer,
    Preprocessor,
    TextTokenizer,
    annotation_label_vocabulary_build,
)


GWSD_DATA_PATH = Path("storage/data/modified/into_json/gwsd_parsed.json")
ANON_DATA_PATH = Path("storage/data/modified/into_json/Anon_parsed.json")

TRAIN_PATH = Path("storage/data/model_data/train/")
VAL_PATH = Path("storage/data/model_data/val/")
TEST_PATH = Path("storage/data/model_data/test/")
UNIQUE_ANN_PATH = Path("storage/data/model_data/unique_annotators/")


def main() -> None:
    """Split datasets + preprocess."""
    dataset_name = DatasetName.gwsd

    # Load the GWSD data from the path for it and create the parsed pydantic instances
    gwsd_raw_instances: list[str] = json.loads(GWSD_DATA_PATH.read_text())
    gwsd_parsed_instances = [
        GwsdSentenceInstance.model_validate_json(instance) for instance in gwsd_raw_instances
    ]
    gwsd_model_instances = [
        ModelInstance.from_gwsd(instances) for instances in gwsd_parsed_instances
    ]

    #  of instances differentiated only by DatasetName and iterate through that.
    gwsd_train_instances, gwsd_val_instances = create_dataset_split(
        instances=gwsd_model_instances, train_size=0.9, val_size=0.1
    )

    vocab = annotation_label_vocabulary_build()
    annotation_tokenizer = AnnotationTokenizer(vocab)

    annotator_metadata = {}
    for instance in gwsd_model_instances:
        for annotator, annotator_info in instance.annotator_info_per_instance.items():
            if annotator not in list(annotator_metadata.keys()):
                annotator_metadata[annotator] = annotator_info

    unique_annotator_id_list = list(annotator_metadata.keys())
    annotator_tokenizer = AnnotatorTokenizer.from_unique_annotators(unique_annotator_id_list)
    Metadata.gwsd_annotator_metadata = annotator_metadata

    text_tokenizer = TextTokenizer()

    # # Instantiate the annotation tokenizer

    # Instantiate the preprocessor
    instance_preprocessor = Preprocessor(
        text_tokenizer=text_tokenizer,
        annotator_tokenizer=annotator_tokenizer,
        annotation_tokenizer=annotation_tokenizer,
    )

    TRAIN_PATH.mkdir(parents=True, exist_ok=True)
    VAL_PATH.mkdir(parents=True, exist_ok=True)
    TEST_PATH.mkdir(parents=True, exist_ok=True)
    UNIQUE_ANN_PATH.mkdir(parents=True, exist_ok=True)
    # Tokenize gwsd instances into text/ annotator seprate instances for encoder / decoder model
    tokenised_separate_train_instances = [
        instance_preprocessor.preprocess_separate(instance) for instance in gwsd_train_instances
    ]

    tokenised_separate_val_instances = [
        instance_preprocessor.preprocess_separate(instance) for instance in gwsd_val_instances
    ]

    with TRAIN_PATH.joinpath(f"{dataset_name.value}_separate_train.pkl").open("wb") as f:
        pickle.dump(tokenised_separate_train_instances, f)

    model_instance_adapter = TypeAdapter(list[ModelInstance])
    TRAIN_PATH.joinpath(f"{dataset_name.value}_train_unencoded.json").write_bytes(
        model_instance_adapter.dump_json(gwsd_train_instances)
    )
    VAL_PATH.joinpath(f"{dataset_name.value}_val_unencoded.json").write_bytes(
        model_instance_adapter.dump_json(gwsd_val_instances)
    )

    with VAL_PATH.joinpath(f"{dataset_name.value}_separate_val.pkl").open("wb") as f:
        pickle.dump(tokenised_separate_val_instances, f)

    with UNIQUE_ANN_PATH.joinpath(f"{dataset_name.value}_annotator_metadata.pkl").open("wb") as f:
        pickle.dump(Metadata.gwsd_annotator_metadata, f)


if __name__ == "__main__":
    main()
