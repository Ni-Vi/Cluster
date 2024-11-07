from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel


S = TypeVar("S", bound=Enum)


class Metadata(BaseModel, Generic[S]):
    """Number of unique annotators per dataset."""

    annotator_metadata: dict[str, S]


class DatasetName(Enum):
    """All possible genders in the dataset."""

    gwsd = "GWSD"
    anon = "Anon"
    babe = "Babe"
    mbic = "MBIC"
    GAB = "gab"
    GWSTANCE = "gwstance"
    IDEOLOGY = "ideology"
