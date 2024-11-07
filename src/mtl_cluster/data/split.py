from collections import Counter
from enum import Enum
from itertools import chain
from typing import TypeVar

import numpy as np
from more_itertools import split_into

from mtl_cluster.data.dataset_instance import ModelInstance


S = TypeVar("S", bound=Enum)
T = TypeVar("T")


def create_dataset_split(
    instances: list[ModelInstance[S, T]],
    train_size: float,
    val_size: float,
) -> tuple[list[ModelInstance[S, T]], list[ModelInstance[S, T]]]:
    """Split the dataset into train, val and test sets."""
    train_list: list[ModelInstance[S, T]] = []
    val_list: list[ModelInstance[S, T]] = []

    annotator_set_count = Counter([instance.annotator_set_hash for instance in instances])
    annotator_set_count = dict(sorted(annotator_set_count.items(), key=lambda x: x[0]))
    annotator_counts = np.array(list(annotator_set_count.values()))
    train_annotations = np.array((annotator_counts * train_size).round().astype(int))
    val_annotations = np.array(np.floor(annotator_counts * val_size).astype(int))

    sanity_check = np.stack((train_annotations, val_annotations), axis=1)

    # needs a more graceful way to handle this but basically fixing the case where for 45
    # annotations per annotator, the train size is 36, val size is 4 and test size is 4 != 44, and I
    # don't want to lose annotations.
    train_size_adjustment = annotator_counts - sanity_check.sum(axis=1)
    sanity_check.T[0] += train_size_adjustment

    assert (sanity_check.sum(axis=1) == annotator_counts).all()

    instances_sorted = sorted(instances, key=lambda x: x.annotator_set_hash)

    split_list = list(split_into(instances_sorted, sanity_check.flatten().tolist()))

    train_list = list(chain(*split_list[::2]))
    val_list = list(chain(*split_list[1::2]))

    return train_list, val_list
