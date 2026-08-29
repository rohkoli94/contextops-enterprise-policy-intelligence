from dataclasses import dataclass


@dataclass(frozen=True)
class SparseEmbedding:
    """
    Provider-neutral sparse vector representation.

    indices:
        IDs of the non-zero dimensions.

    values:
        Values corresponding to those dimensions.
    """

    indices: list[int]
    values: list[float]