# serving/registry/base.py
from abc import ABC, abstractmethod

class ModelVersion(ABC):
    @abstractmethod
    def predict(self, raw_input) -> tuple[dict, float]:
        """Returns (output_dict, confidence). Subclass owns its own preprocessing."""
        ...

    @abstractmethod
    def embed(self, raw_input):
        """Returns a reference-space vector/features for drift comparison (Phase 6)."""
        ...