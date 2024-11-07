import itertools
from typing import Protocol, Self

import torch
from einops import rearrange
from sentence_transformers.models import Pooling
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    GPT2Model,
    PreTrainedModel,
    T5Model,
)


class DownsampleEmbedding(torch.nn.Sequential):
    @classmethod
    def from_params(
        cls,
        starting_dim: int,
        end_dim: int,
        num_layers: int,
        dropout: float,
        *,
        use_layer_norm: bool = True,
    ) -> Self:
        # If we want no downsampling, we just return an identity layer
        if num_layers == 0:
            return cls(torch.nn.Identity())

        assert starting_dim > end_dim, "Starting dimension must be larger than end dimension"

        step = (starting_dim - end_dim) // num_layers

        intermediate_dims = [starting_dim - step * i for i in range(num_layers + 1)]
        dim_pairs = list(itertools.pairwise(intermediate_dims))

        assert (
            intermediate_dims[0] == starting_dim
        ), "First dimension must be equal to starting dimension"
        assert intermediate_dims[-1] == end_dim, "Last dimension must be equal to end dimension"
        assert (
            len(dim_pairs) == num_layers
        ), "Number of layers must be equal to number of dimension pairs"

        modules = []
        for start_dim, end_dim in dim_pairs:
            modules.append(
                torch.nn.Linear(start_dim, end_dim),
            )
            modules.append(torch.nn.GELU())
            if use_layer_norm:
                modules.append(torch.nn.LayerNorm(end_dim))
            modules.append(torch.nn.Dropout(dropout))

        return cls(*modules)


class ModelProtocol(Protocol):
    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor | None = None,
        annotator_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...

    def decode_annotator_ids(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor | None = None,
        annotator_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


class CrossAttentionModel(torch.nn.Module, ModelProtocol):
    """Encoder-Decoder model."""

    def __init__(
        self,
        *,
        encoder: PreTrainedModel,
        unique_annotators: int,
        annotation_vocab_size: int,
        text_dim: int = 768,
        ann_dim: int = 768,
        downsample_num_layers: int = 0,
        modalities: int = 4,
        decoder_depth: int = 12,
        padding_tokens: list[int] = [],
        dropout: float = 0,
        disable_transformer_ff_bias: bool = False,
        disable_projection_bias: bool = False,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self.unique_annotators = unique_annotators

        self.ann_dim = ann_dim
        self.num_heads = modalities
        self.padding_tokens = padding_tokens

        self.downsampling = DownsampleEmbedding.from_params(
            starting_dim=text_dim,
            end_dim=ann_dim,
            num_layers=downsample_num_layers,
            dropout=dropout,
        )

        # When using datasets with non-equal number of annotators per instance, such that we
        # can have mixed batches, we need to address this so that we account accordingly.
        self.ann_embedding_layer = torch.nn.Embedding(
            self.unique_annotators + 1,  # Plus one to account for padding
            ann_dim,
            padding_idx=0,
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.ann_dim,
            nhead=modalities,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=dropout,
            bias=not disable_transformer_ff_bias,
        )
        self.annotation_vocab_size = annotation_vocab_size
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.projection = torch.nn.Linear(
            ann_dim, annotation_vocab_size, bias=not disable_projection_bias
        )

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        encoded_annotator_id = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        logits = self.predict_logits(encoded_annotator_id)
        return logits

    def decode_annotator_ids(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Inverting mask since huggingface works opposite to our convention. That's why we use ~.
        encoded_text = self._encoder(
            input_ids=text_tokens, attention_mask=~text_mask
        ).last_hidden_state
        encoded_text = self.downsampling(encoded_text)

        # encoded_text = self._encoder(
        #     input_ids=text_tokens, attention_mask=~text_mask
        # ).pooler_output.
        ann_embeddings = self.ann_embedding_layer(annotator_tokens)

        encoded_annotator_id = self.decoder(
            tgt=ann_embeddings,
            memory=encoded_text,
            tgt_key_padding_mask=annotator_mask,
            memory_key_padding_mask=text_mask,
        )
        # Shape (batch, num annotators, ann_dim)
        return encoded_annotator_id

    def predict_logits(self, encoded_annotator_ids: torch.Tensor) -> torch.Tensor:
        """Predict the logits for the annotator tokens."""
        logits = self.projection(encoded_annotator_ids)
        return logits


class CrossAttentionPooledModel(CrossAttentionModel):
    """Cross-Attention model with pooled annotator embeddings."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling = Pooling(self.ann_dim, pooling_mode="mean")

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        encoded_annotator_id = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        pooled_ids = self.pooling(
            {"token_embeddings": encoded_annotator_id, "attention_mask": annotator_mask}
        )["sentence_embedding"]
        logits = self.predict_logits(pooled_ids)
        return logits


class SequenceClassificationModel(torch.nn.Module, ModelProtocol):
    """Encoder only model."""

    def __init__(self, *, encoder: PreTrainedModel) -> None:
        super().__init__()
        self._encoder = encoder

        if encoder.config.model_type in {"bert", "deberta-v2"}:
            self._classification_head = self._encoder.classifier
        if encoder.config.model_type in {"roberta"}:
            self._classification_head = self._encoder.classifier.out_proj
        if encoder.config.model_type == "t5":
            self._classification_head = self._encoder.classification_head

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, unique_annotators: int, annotation_vocab_size: int
    ) -> Self:
        """Create a new model instance given a pretrained model name or path."""
        encoder = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            config=AutoConfig.from_pretrained(
                pretrained_model_name_or_path, num_labels=annotation_vocab_size
            ),
        )
        assert isinstance(encoder, PreTrainedModel)
        encoder.resize_token_embeddings(encoder.config.vocab_size + unique_annotators)

        return cls(encoder=encoder)

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor | None = None,  # noqa: ARG002
        annotator_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass of the model."""
        encoded_text = self._encoder(input_ids=text_tokens, attention_mask=~text_mask).logits
        return encoded_text

    def decode_annotator_ids(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor | None = None,  # noqa: ARG002
        annotator_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Decode the annotator ids."""
        # TODO: Check this
        sentence_representation_list = []

        hook = self._classification_head.register_forward_hook(
            lambda _module, args, _output: sentence_representation_list.append(args)
        )
        _ = self.forward(text_tokens=text_tokens, text_mask=text_mask)
        hook.remove()

        # Shape (batch, ann_dim)
        latent_space = sentence_representation_list[0][0]
        latent_space = rearrange(latent_space, "batch ann_dim -> batch 1 ann_dim")
        return latent_space


class EncoderEncoderModel(torch.nn.Module, ModelProtocol):
    """Decoder only model."""

    def __init__(
        self,
        *,
        encoder: PreTrainedModel,
        unique_annotators: int,
        annotation_vocab_size: int,
        text_dim: int = 768,
        ann_dim: int = 768,
        downsample_num_layers: int = 0,
        modalities: int = 4,
        decoder_depth: int = 12,
        padding_tokens: list[int] = [],
        dropout: float = 0,
        disable_transformer_ff_bias: bool = False,
        disable_projection_bias: bool = False,
    ) -> None:
        super().__init__()
        self._encoder = encoder
        self.unique_annotators = unique_annotators
        self.ann_dim = ann_dim
        self.num_heads = modalities
        self.padding_tokens = padding_tokens
        self.downsampling = DownsampleEmbedding.from_params(
            starting_dim=text_dim,
            end_dim=ann_dim,
            num_layers=downsample_num_layers,
            dropout=dropout,
        )

        # When using datasets with non-equal number of annotators per instance, such that we
        # can have mixed batches, we need to address this so that we account accordingly.
        self.ann_embedding_layer = torch.nn.Embedding(
            self.unique_annotators + 1,  # Plus one to account for padding
            ann_dim,
            padding_idx=0,
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.ann_dim,
            nhead=modalities,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=dropout,
            bias=not disable_transformer_ff_bias,
        )
        self.annotation_vocab_size = annotation_vocab_size
        self.decoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=decoder_depth)

        self.projection = torch.nn.Linear(
            ann_dim, annotation_vocab_size, bias=not disable_projection_bias
        )

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        encoded_annotator_id = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        logits = self.predict_logits(encoded_annotator_id)
        return logits

    def get_embedded_padding(self, device: torch.device) -> torch.Tensor:
        """Get the embedded padding tokens."""
        # TODO: rename padding to separator.butchering terms
        padding = torch.tensor(self.padding_tokens, device=device).unsqueeze(0)
        encoded_padding = self._encoder(input_ids=padding).last_hidden_state
        return encoded_padding

    def decode_annotator_ids(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Inverting mask since huggingface works opposite to our convention. That's why we use ~.
        encoded_text = self._encoder(
            input_ids=text_tokens, attention_mask=~text_mask
        ).last_hidden_state

        encoded_text = self.downsampling(encoded_text)

        ann_embeddings = self.ann_embedding_layer(annotator_tokens)
        padding = self.get_embedded_padding(device=text_tokens.device).repeat(
            encoded_text.size(0), 1, 1
        )
        padding_mask = padding.new_zeros(padding.size(0), padding.size(1)).bool()

        text_annotators = torch.cat([encoded_text, padding, ann_embeddings], dim=1)

        text_annotators_masks = torch.cat([text_mask, padding_mask, annotator_mask], dim=1)

        decoder_pass = self.decoder(
            src=text_annotators,
            src_key_padding_mask=text_annotators_masks,
        )

        encoded_annotator_id = decoder_pass[:, -ann_embeddings.size(1) :]
        # Shape (batch, num annotators, ann_dim)
        return encoded_annotator_id

    def predict_logits(self, encoded_annotator_ids: torch.Tensor) -> torch.Tensor:
        """Predict the logits for the annotator tokens."""
        logits = self.projection(encoded_annotator_ids)
        return logits


class EncoderEncoderPooledModel(EncoderEncoderModel):
    """EncoderEncoder model with pooled annotator embeddings."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling = Pooling(self.ann_dim, pooling_mode="mean")

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        encoded_annotator_id = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        pooled_ids = self.pooling(
            {"token_embeddings": encoded_annotator_id, "attention_mask": annotator_mask}
        )["sentence_embedding"]
        logits = self.predict_logits(pooled_ids)
        return logits


class DecoderOnlyModel(torch.nn.Module, ModelProtocol):
    """Decoder only model."""

    def __init__(
        self,
        *,
        decoder: PreTrainedModel,
        annotation_vocab_size: int,
        disable_projection_bias: bool = False,
        separator_tokens: list[int] = [],
    ) -> None:
        super().__init__()
        self._decoder = decoder
        self.annotation_vocab_size = annotation_vocab_size
        self.separator_tokens = separator_tokens
        self.projection = torch.nn.Linear(
            self._decoder.config.n_embd, annotation_vocab_size, bias=not disable_projection_bias
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        unique_annotators: int,
        annotation_vocab_size: int,
        *,
        disable_projection_bias: bool = False,
        separator_tokens: list[int] = [],
    ) -> Self:
        """Create a decoder model instance given a pretrained model / path."""
        decoder = GPT2Model.from_pretrained(
            pretrained_model_name_or_path,
            config=AutoConfig.from_pretrained(
                pretrained_model_name_or_path, num_labels=annotation_vocab_size
            ),
        )
        assert isinstance(decoder, PreTrainedModel)
        decoder.resize_token_embeddings(decoder.config.vocab_size + unique_annotators)

        return cls(
            decoder=decoder,
            annotation_vocab_size=annotation_vocab_size,
            disable_projection_bias=disable_projection_bias,
            separator_tokens=separator_tokens,
        )

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor | None = None,
        annotator_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        decoded_text = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        logits = self.predict_logits(decoded_text)
        return logits

    def predict_logits(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Predict the logits for the annotator tokens."""
        logits = self.projection(latent_states)
        return logits

    def decode_annotator_ids(
        self,
        *,
        # shape (batch_size, seqlen)
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        # shape (batch_size, annotator_seq_len)
        annotator_tokens: torch.Tensor | None = None,
        annotator_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode the annotator ids."""
        # shape (Batch_size, sep_len = 1)
        separator = (
            torch.tensor(self.separator_tokens, device=text_tokens.device)
            .unsqueeze(0)
            .repeat(text_tokens.size(0), 1)
        )
        # shape (Batch_size, mask_sep_len = 1)
        separator_mask = separator.new_zeros(separator.size(0), separator.size(1)).bool()

        assert annotator_tokens is not None
        assert annotator_mask is not None

        # shape (batch_size, seq_len + sep_len + annotator_seq_len)
        text_annotators = torch.cat([text_tokens, separator, annotator_tokens], dim=1)

        # shape (batch_size, seq_len + sep_len + annotator_seq_len)
        text_annotators_masks = torch.cat([text_mask, separator_mask, annotator_mask], dim=1)

        # shape (batch_size, seq_len + sep_len + annotator_seq_len, hidden_size)
        decoded_inputs = self._decoder(
            input_ids=text_annotators, attention_mask=~text_annotators_masks
        ).last_hidden_state

        # hape = (batch_size, annotator seq_len, hidden_size)
        decoded_input_ids = decoded_inputs[:, -annotator_tokens.size(1) :]
        return decoded_input_ids


class EncoderDecoderPretrainedModel(torch.nn.Module, ModelProtocol):
    """Encoder-Decoder model with pretrained encoder and decoder."""

    def __init__(
        self,
        *,
        encoder_decoder: PreTrainedModel,
        unique_annotators: int,
        annotation_vocab_size: int,
        disable_projection_bias: bool = False,
    ) -> None:
        super().__init__()
        self._encoder_decoder = encoder_decoder
        self.unique_annotators = unique_annotators

        self.projection = torch.nn.Linear(
            self._encoder_decoder.config.d_model,
            annotation_vocab_size,
            bias=not disable_projection_bias,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, unique_annotators: int, annotation_vocab_size: int
    ) -> Self:
        """Create a new model instance given a pretrained model name or path."""
        encoder_decoder = T5Model.from_pretrained(
            pretrained_model_name_or_path,
            config=AutoConfig.from_pretrained(
                pretrained_model_name_or_path, num_labels=annotation_vocab_size
            ),
        )
        assert isinstance(encoder_decoder, PreTrainedModel)
        encoder_decoder.resize_token_embeddings(
            encoder_decoder.config.vocab_size + unique_annotators
        )

        return cls(
            encoder_decoder=encoder_decoder,
            unique_annotators=unique_annotators,
            annotation_vocab_size=annotation_vocab_size,
        )

    def forward(
        self,
        *,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded_annotator_id = self.decode_annotator_ids(
            text_tokens=text_tokens,
            text_mask=text_mask,
            annotator_tokens=annotator_tokens,
            annotator_mask=annotator_mask,
        )
        logits = self.predict_logits(encoded_annotator_id)
        return logits

    def decode_annotator_ids(
        self,
        *,
        # shape (batch_size, seq_len)
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        # shape (batch_size, ann_seq_len)
        annotator_tokens: torch.Tensor,
        annotator_mask: torch.Tensor,
    ) -> torch.Tensor:
        # shape (batch_size, seq_len, hidden_size)
        cross_attention_output = self._encoder_decoder(
            input_ids=text_tokens,
            attention_mask=~text_mask,
            decoder_input_ids=annotator_tokens,
            decoder_attention_mask=~annotator_mask,
        ).last_hidden_state

        # TODO: remove this?

        return cross_attention_output

    def predict_logits(self, encoded_annotator_ids: torch.Tensor) -> torch.Tensor:
        """Predict the logits for the annotator tokens."""
        logits = self.projection(encoded_annotator_ids)
        return logits
