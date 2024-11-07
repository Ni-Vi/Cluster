from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class GabHateCorpusAnnotationsInstance(BaseModel):
    """"Single instance of the GAB Labeled Dataset.
    """

    ID : int
    Annotator : int
    text : str
    topic : Literal["environment", "abortion","immigration","international-politics-and-world-news","gun-control","white-nationalism", "vaccines", "sport", "middle-class", "gender", "elections-2020", "trump-presidency" , "student-debt"]
    type : Literal["left","center","right"]
    Hate : int
    HD : Literal[0,1]
    CV : Literal[0,1]
    VO : Literal[0,1]
    REL : int | None
    RAE : int | None
    SXO : int | None
    GEN : int | None
    IDL : int | None
    NAT : int | None
    POL : int | None
    MPH : int | None
    EX : int | None
    IM : int | None


class RawLabelsMBIC(BaseModel):
    """"Single instance of the GAB AnnotatorAttitudes dataset.

    """
    Annotator : int
    PostsAnnotated : int
    IAT-RACE : float
    IAT-GenderCareer : int
    IAT-Sexuality :	float
    IAT-Religion : float
    HCBS-NegativeBelief : float
    HCBS-OffenderPunishment : float
    HCBS-Deterrence : float
    HCBS-VictimHarm : float


storage_dir = Path("storage/")

anonymous_dataset = storage_dir.joinpath("BABE/")
