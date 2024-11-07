import json
from pathlib import Path

from transformers import AutoTokenizer

from mtl_cluster.data.datasets.anonymous_into_json import AnonDatasetInstance
from mtl_cluster.modules.preprocess_instances import ModelInstance
from mtl_cluster.modules.tokenisers import AnnotationTokenizer


def test_load_anon_dataset() -> None:
    anon_input_file = Path("storage/data/modified/into_json/Anon_parsed.json")

    anon_raw_instances: list[str] = json.loads(anon_input_file.read_text())
    anon_parsed_instances = [
        AnonDatasetInstance.model_validate_json(instance) for instance in anon_raw_instances
    ]

    anon_output_instances = [
        ModelInstance.from_anon(instances) for instances in anon_parsed_instances
    ]
    assert anon_output_instances


def run():
    annotation_tokenizer = AnnotationTokenizer(vocab=build_vocab())
    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    preprocessor = Preprocessor(
        text_tokenizer=text_tokenizer, annotation_tokenizer=annotation_tokenizer
    )
