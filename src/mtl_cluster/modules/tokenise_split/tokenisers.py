from enum import Enum
from typing import Any, Self, TypeVar

import torch
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

from mtl_cluster.data.dataset_instance import ModelInstance
from mtl_cluster.data.datasets.anonymous_into_json import BiasLabel
from mtl_cluster.data.datasets.gwsd_into_json import AnnotationValue


ANNOTATION_LABEL_LIST: list[type[Enum]] = [AnnotationValue, BiasLabel]

S = TypeVar("S", bound=Enum)
T = TypeVar("T")


def annotation_label_vocabulary_build(
    annotation_label: list[type[Enum]] = ANNOTATION_LABEL_LIST,
) -> dict[str, int]:
    """Creates a vocabulary that is consistent across all datasets."""
    annotations_string: set[str] = {
        item.name for annotation in annotation_label for item in annotation
    }
    annotations_dict: dict[str, int] = {
        item: index for index, item in enumerate(sorted(annotations_string))
    }
    return annotations_dict


def get_annotation_vocab_size(
    annotation_label: list[type[Enum]] = ANNOTATION_LABEL_LIST,
) -> int:
    """Gets the size of the vocabulary."""
    return len(annotation_label_vocabulary_build(annotation_label))


NUM_NEW_ANNOTATORS = 10000
# TODO: FIX NUMBER OF ANNOTATORS TO BE CORRECT


def create_sentence(instance: ModelInstance) -> str:
    """Creates sentence into str for the model."""
    return f"{instance.sentence}"


class AnnotationTokenizer:
    """Takes annotation, changes into int."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab: dict[str, int] = vocab

    def encode(self, annotation: list[Enum]) -> list[int]:
        """Converts annotation into integer."""
        return [self.vocab[ann_value.name] for ann_value in annotation]

        # annotations = [annotations.name for annotations in instance.annotations]


class TextTokenizer:
    """Tokenises prompt text."""

    def __init__(
        self,
        pretrained_model: str = "google/t5-v1_1-large",
        **kwargs: Any,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **kwargs)


class AnnotatorTokenizer:
    """Tokenizes annotators."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab: dict[str, int] = vocab

    @classmethod
    def from_unique_annotators(cls, unique_annotator_ids: list[str]) -> Self:
        """Create a vocabulary based on the number of annotators.

        We +1 to the indexes to account for padded annotators. Padding index is set to 0 within the
        model, so we need to make sure that the annotator indexes start from 1.
        """
        annotator_special_tokens = {
            f"ann_{ann_id}": idx + 1 for idx, ann_id in enumerate(sorted(unique_annotator_ids))
        }
        return cls(vocab=annotator_special_tokens)

    def encode(self, annotator: list[str]) -> list[int]:
        """Converts annotation into integer."""
        return [self.vocab[f"ann_{ann_id}"] for ann_id in annotator]


def create_annotators(instance: ModelInstance) -> list[int]:
    """Create instances of annotators to be tokenised."""
    list_of_annotators = [int(annotator) for annotator in instance.annotators]
    return list_of_annotators


class PreprocessedInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: torch.Tensor
    annotators: list[int] | None
    annotations: list[int] | None


class Preprocessor:
    """Preprocesses the data."""

    def __init__(
        self,
        text_tokenizer: TextTokenizer,
        annotation_tokenizer: AnnotationTokenizer,
        annotator_tokenizer: AnnotatorTokenizer,
    ) -> None:
        self.text_tokenizer: TextTokenizer = text_tokenizer
        self.annotation_tokenizer: AnnotationTokenizer = annotation_tokenizer
        self.annotator_tokenizer: AnnotatorTokenizer = annotator_tokenizer

    def preprocess_separate(self, instance: ModelInstance) -> PreprocessedInstance:
        """Tokenizes the instance as a sentence and annotator seperate."""
        # Since we are not batching, nor using more than one SEP token, we do not care about the
        # token_type_ids or the attention masks. Therefore, we just keep the token IDs.

        dataset_name = instance.dataset_name.name
        # TODO: use dataset_names if more than one dataset
        sentence_ = create_sentence(instance)
        # annotators_ = create_annotators(instance)

        tokenized_sentence = self.text_tokenizer.tokenizer(sentence_, return_tensors="pt")

        annotator_tokens = self.annotator_tokenizer.encode(instance.annotators)

        annotation_tokens = self.annotation_tokenizer.encode(instance.annotations)

        text_tokens = tokenized_sentence["input_ids"]

        assert isinstance(text_tokens, torch.Tensor)

        return PreprocessedInstance(
            text=text_tokens, annotators=annotator_tokens, annotations=annotation_tokens
        )
