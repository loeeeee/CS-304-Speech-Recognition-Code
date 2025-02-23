from dataclasses import dataclass, field
from typing import Dict, List, Self, Tuple
import logging

from numpy.typing import NDArray
import numpy as np

from loe_speech_recognition import TI_DIGITS_LABEL_TYPE

logger = logging.getLogger(__name__)


@dataclass
class HiddenMarkovModelInference:
    # Internal
    _label_locations: Dict[Tuple[int, int], TI_DIGITS_LABEL_TYPE] = field(default_factory=dict)
    _means: NDArray[np.float32] = field(init=False)
    _covariances: NDArray[np.float32] = field(init=False)
    _transition_probs: NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        logger.debug(f"An HMM inference object created")

    @classmethod
    def load_from_folder(cls, folder_path: str, models_to_load: List[str]) -> Self:

        ...

    def predict(self, signal: NDArray[np.float32]) -> str:
        ...

    @staticmethod
    def model_folder_name_parser(folder_path: str) -> Tuple[int, int, int]:
        """
        Parse folder name of each model

        Args:
            folder_path (str): Full path of the folder

        Returns:
            Tuple[int, int, int]: Label, Number of states, Dimension of features
        """
        
        ...
