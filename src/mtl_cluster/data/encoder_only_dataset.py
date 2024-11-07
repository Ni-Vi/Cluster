from itertools import chain

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from mtl_cluster.data.dataset_instance import ModelInstance
from mtl_cluster.modules.tokenise_split.tokenisers import AnnotationTokenizer, PreprocessedInstance


def convert_annotator_to_placeholder(annotator: str) -> str:
    """Convert the annotator name to a placeholder."""
    return f"ann_{annotator}"


# def add_tokens_to_tokenizer(
#     tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, instances: list[ModelInstance]
# ) -> int:
#     """Add the annotator tokens to the tokenizer."""
#     all_annotators = (annotator for instance in instances for annotator in instance.annotators)
#     annotator_tokens = {
#         convert_annotator_to_placeholder(annotator) for annotator in all_annotators
#     }
#     sorted_tokens = sorted(annotator_tokens)
#     num_new_tokens = tokenizer.add_tokens(list(sorted_tokens))
#     return num_new_tokens


def prepare_instances_for_encoder_only_model(
    instances: list[ModelInstance],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> list[PreprocessedInstance]:
    """Prepare the instances for the encoder-only model.

    This is because we are doing things differently.
    """
    preprocessed_instances: list[PreprocessedInstance] = []

    for instance in tqdm(instances, desc="Preprocessing instances"):
        for annotator, annotation in zip(instance.annotators, instance.annotations, strict=True):
            new_sentence = f"{instance.sentence} // {convert_annotator_to_placeholder(annotator)}"
            tokenized_text = tokenizer(new_sentence, return_tensors="pt").input_ids
            tokenized_annotation = annotation_tokenizer.encode([annotation])
            tokenized_annotator = tokenizer(
                convert_annotator_to_placeholder(annotator),
                add_special_tokens=False,
            ).input_ids  # tokenized_annotator: tensor(20527)
            assert len(tokenized_annotator) == 1
            preprocessed_instances.append(
                PreprocessedInstance(
                    text=tokenized_text,
                    annotators=tokenized_annotator,
                    annotations=tokenized_annotation,
                )
            )

    return preprocessed_instances


def prepare_instances_for_decoder_only_model(
    instances: list[ModelInstance],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> list[PreprocessedInstance]:
    """Prepare the instances for the decoder-only model."""
    preprocessed_instances: list[PreprocessedInstance] = []

    for instance in tqdm(instances, desc="Preprocessing instances"):
        instance_annotators = [
            convert_annotator_to_placeholder(annotator) for annotator in instance.annotators
        ]
        instance_annotations = [
            annotation_tokenizer.encode([annotation]) for annotation in instance.annotations
        ]
        flattened_list = list(chain.from_iterable(instance_annotations))
        tokenized_annotators = tokenizer(instance_annotators, return_tensors="pt").input_ids
        tokenized_text = tokenizer(instance.sentence, return_tensors="pt").input_ids
        tokenized_annotators_without_eos = tokenized_annotators[:, 0]
        preprocessed_instances.append(
            PreprocessedInstance(
                text=tokenized_text,
                annotators=tokenized_annotators_without_eos,
                annotations=flattened_list,
            )
        )

    return preprocessed_instances


def prepare_instances_for_encoder_decoder_pretrained_model(
    instances: list[ModelInstance],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    annotation_tokenizer: AnnotationTokenizer,
) -> list[PreprocessedInstance]:
    """Prepare the instances for the decoder-only model."""
    preprocessed_instances: list[PreprocessedInstance] = []

    for instance in tqdm(instances, desc="Preprocessing instances"):
        instance_annotators = [
            convert_annotator_to_placeholder(annotator) for annotator in instance.annotators
        ]
        instance_annotations = [
            annotation_tokenizer.encode([annotation]) for annotation in instance.annotations
        ]
        flattened_list = list(chain.from_iterable(instance_annotations))
        tokenized_annotators = tokenizer(instance_annotators, return_tensors="pt").input_ids
        tokenized_annotators_without_eos = tokenized_annotators[:, 0]
        tokenized_text = tokenizer(instance.sentence, return_tensors="pt").input_ids
        preprocessed_instances.append(
            PreprocessedInstance(
                text=tokenized_text,
                annotators=tokenized_annotators_without_eos,
                annotations=flattened_list,
            )
        )

    return preprocessed_instances
