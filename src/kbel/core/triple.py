from dataclasses import dataclass


@dataclass
class Triple:
    """A subject-predicate-object triple extracted from text."""

    subject: str
    predicate: str
    object: str