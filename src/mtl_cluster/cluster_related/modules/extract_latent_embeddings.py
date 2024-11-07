import itertools

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from mtl_cluster.cluster_related.data_structures.cluster_input import ClusteringInput
from mtl_cluster.data.datamodule import DatasetOutput, MTLDataModule
from mtl_cluster.lightning_modules.cluster import ClusterLightningModule
from mtl_cluster.modules.tokenise_split.tokenisers import AnnotatorTokenizer, TextTokenizer


def collate_latent_spaces_custom_model(
    args: list[tuple[DatasetOutput, torch.Tensor]],
    *,
    pretrained_model: str,
    annotator_tokenizer: AnnotatorTokenizer,
) -> ClusteringInput:
    """Get the latent spaces and the annotator ids together in preparation for clustering."""
    text_tokenizer = TextTokenizer(pretrained_model=pretrained_model)
    _annotator_tokenizer = annotator_tokenizer

    latent_spaces = []
    annotators_list = []
    annotations_list = []
    text_list = []

    for batch, latent_space in args:
        assert batch.annotations is not None
        assert batch.annotations_mask is not None
        assert torch.all(batch.annotations_mask == batch.annotators_mask)

        mask = ~batch.annotators_mask.view(-1)
        annotations = batch.annotations.view(-1)[mask]
        other_latent_space = latent_space.contiguous().view(-1, latent_space.size(-1))[mask]

        decoded_text_batch: list[str] = text_tokenizer.tokenizer.batch_decode(
            batch.text, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        texts_per_annotator: list[int] = (~batch.annotations_mask).sum(dim=-1).tolist()

        repeated_texts = list(
            itertools.chain.from_iterable(
                [
                    itertools.repeat(repeated_text, num_repeats)
                    for repeated_text, num_repeats in zip(
                        decoded_text_batch, texts_per_annotator, strict=True
                    )
                ]
            )
        )
        batch_annotators_decoded = [
            key
            for annotator in torch.flatten(batch.annotators)
            for key, value in annotator_tokenizer.vocab.items()
            if value == annotator
        ]
        annotator_ids = [int(ann.replace("ann_", "")) for ann in batch_annotators_decoded]

        text_list.extend(repeated_texts)
        latent_spaces.append(other_latent_space)
        annotations_list.extend(annotations.tolist())
        annotators_list.extend(annotator_ids)

    latent_space_tensor = torch.cat(latent_spaces, dim=0)
    return ClusteringInput(
        annotator_ids=annotators_list,
        annotation_ids=annotations_list,
        text=text_list,
        embedding_space=latent_space_tensor,
    )


def collate_latent_spaces_pt_model(
    args: list[tuple[DatasetOutput, torch.Tensor]],
    *,
    pretrained_tokenizer: PreTrainedTokenizerBase,
) -> ClusteringInput:
    """Get the latent spaces and the annotator ids together in preparation for clustering."""
    latent_spaces = []
    annotators_list = []
    annotations_list = []
    text_list = []

    for batch, latent_space in args:
        assert batch.annotations is not None
        assert batch.annotations_mask is not None
        assert torch.all(batch.annotations_mask == batch.annotators_mask)

        mask = ~batch.annotators_mask.view(-1)
        annotations = batch.annotations.view(-1)[mask]

        other_latent_space = latent_space.contiguous().view(-1, latent_space.size(-1))[mask]

        decoded_text_batch: list[str] = pretrained_tokenizer.batch_decode(
            batch.text, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        texts_per_annotator: list[int] = (~batch.annotations_mask).sum(dim=-1).tolist()

        repeated_texts = list(
            itertools.chain.from_iterable(
                [
                    itertools.repeat(repeated_text, num_repeats)
                    for repeated_text, num_repeats in zip(
                        decoded_text_batch, texts_per_annotator, strict=True
                    )
                ]
            )
        )
        annotators_per_text = pretrained_tokenizer.batch_decode(
            batch.annotators.view(-1)[batch.annotators.view(-1) != 0]
        )
        annotator_ids = [int(ann.replace("ann_", "")) for ann in annotators_per_text]
        text_list.extend(repeated_texts)
        latent_spaces.append(other_latent_space)
        annotations_list.extend(annotations.tolist())
        annotators_list.extend(annotator_ids)

    latent_space_tensor = torch.cat(latent_spaces, dim=0)
    return ClusteringInput(
        annotator_ids=annotators_list,
        annotation_ids=annotations_list,
        text=text_list,
        embedding_space=latent_space_tensor,
    )


def extract_model_predictions(
    datamodule: MTLDataModule, lightning_module: ClusterLightningModule
) -> list[tuple[DatasetOutput, torch.Tensor]]:
    """Extract the model predictions."""
    model_predictions = []
    with torch.inference_mode():
        for batch in datamodule.val_dataloader():
            latent_space = lightning_module.predict_step(batch, batch_idx=0, dataloader_idx=0)
            model_predictions.append((batch, latent_space))

    return model_predictions


def datamodule_to_embeddings_val_pt_model(
    datamodule: MTLDataModule,
    lightning_module: ClusterLightningModule,
    *,
    pretrained_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> ClusteringInput:
    """Extract a table of embeddings & annotator id from the validation dataloader."""
    model_predictions = extract_model_predictions(datamodule, lightning_module)
    all_latent_embeddings = collate_latent_spaces_pt_model(
        model_predictions, pretrained_tokenizer=pretrained_tokenizer
    )

    return all_latent_embeddings


def datamodule_to_embeddings_val_custom_model(
    datamodule: MTLDataModule,
    lightning_module: ClusterLightningModule,
    *,
    pretrained_model: str,
    annotator_tokenizer: AnnotatorTokenizer,
) -> ClusteringInput:
    """Extract a table of embeddings & annotator id from the validation dataloader."""
    model_predictions = extract_model_predictions(datamodule, lightning_module)
    all_latent_embeddings = collate_latent_spaces_custom_model(
        model_predictions,
        pretrained_model=pretrained_model,
        annotator_tokenizer=annotator_tokenizer,
    )

    return all_latent_embeddings


# def datamodule_to_embeddings_train(
#     datamodule: MTLDataModule, lightning_module: ClusterLightningModule, *, pretrained_model: str
# ) -> ClusteringInput:
#     """Extract a table of embeddings & annotator id from the validation dataloader."""
#     model_predictions = []
#     with torch.inference_mode():
#         for batch in datamodule.train_dataloader():
#             latent_space = lightning_module.predict_step(batch, batch_idx=0, dataloader_idx=0)
#             model_predictions.append((batch, latent_space))

#     all_latent_embeddings = collate_latent_spaces(
#         model_predictions, pretrained_model=pretrained_model
#     )

#     return all_latent_embeddings
