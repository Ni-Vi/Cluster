from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Gender(Enum):
    """gender of annotator."""

    male = "Male"
    female = "Female"
    other = "Other/Prefer not to say"


class Education(Enum):
    """all possible education levels in the dataset."""

    bachelors = "Bachelors degree"
    vocational = "Vocational or technical school"
    associate_degree = "Associate degree"
    graduate = "Graduate work"
    college_degree = "Some college"
    highschool_graduate = "High school graduate"
    some_highschool = "Some high school"
    undisclosed = "I prefer not to say"


class Nativespeaker(Enum):
    """Whether the person is a native english speaker or not."""

    native_speaker = "Native speaker"
    near_native = "Near-native speaker"
    non_native = "Non-native speaker"


class BiasLabel(Enum):
    """Coding of agreement / disagreement / neutral stance."""

    non_biased = "Non-biased"
    biased = "Biased"



class RawSG2DatasetInstance(BaseModel):
    """"Single instance of the SG1 Labeled Dataset.
    """

    sentence : str
    news_link : HttpUrl
    outlet : Literal[ "msnbc", "usa-today", "breitbart", "federalist", "huffpost" , "fox-news"]
    topic : Literal["environment", "abortion","immigration","international-politics-and-world-news","gun-control","white-nationalism", "vaccines", "sport", "middle-class", "gender", "elections-2020", "trump-presidency" , "student-debt"]
    type :	Literal["left","center","right"]
    group_id :	int
    num_sent :	int
    Label_bias : Literal["Biased","Non-biased"]
    Label_opinion :	Literal["Entirely factual","No agreement","Expresses wleter`s opinion","Somewhat factual but also opinionated", ""] # what about empty?
    biased_words	: str | None
    label_bias_0-1: Literal[0,1]
    annotator_id: list[0:10]



storage_dir = Path("storage/")

anonymous_dataset = storage_dir.joinpath("BABE/")
