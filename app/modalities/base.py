from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseModalityAnalyzer(ABC):
    """
    Abstract Base Class for analysis modalities (Facial, Voice, etc.).
    Enforces a consistent interface so new modalities can be plugged into
    the pipeline without altering API endpoints or response contracts.
    """

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Returns the identifier name of the modality (e.g. 'face', 'voice')."""
        pass

    @abstractmethod
    async def analyze(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes analysis for this modality.
        
        Returns a dictionary containing:
          - scores: Dict[str, Dict[str, float]] -> {'depression': {'value': float, 'confidence': float}, ...}
          - features: List[str] -> list of top contributing descriptors
          - metadata: Dict[str, Any]
        """
        pass
