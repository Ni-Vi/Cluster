import pyarrow as pa
import torch
from pydantic import BaseModel, ConfigDict


class ClusteringInput(BaseModel):
    """Output of the dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    annotator_ids: list[int]
    text: list[str]
    annotation_ids: list[int]
    embedding_space: torch.Tensor

    def to_pyarrow_table(self) -> pa.Table:
        """Convert this into a table."""
        return pa.table(
            {
                "annotator_ids": pa.array(self.annotator_ids),
                "text": pa.array(self.text),
                "annotation_ids": pa.array(self.annotation_ids),
                "embedding_space": pa.array(list(self.embedding_space.numpy())),
            }
        )
