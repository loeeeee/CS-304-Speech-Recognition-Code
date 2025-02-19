from dataclasses import dataclass, field
import os
from typing import Dict, List, Literal, Self
import logging

import numpy as np
from numpy.typing import NDArray

from .hidden_markov_model import HiddenMarkovModel
from .ti_digits import TI_DIGITS_LABEL_TYPE, TI_DIGITS_LABELS

logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class ModelCollection:
    # Mains
    num_of_states: int = field(default=5)
    dim_of_feature: int = field(default=39)

    # Internals
    _models: Dict[TI_DIGITS_LABEL_TYPE, HiddenMarkovModel] = field(default_factory=dict)

    def predict(self, signal: NDArray[np.float32]) -> TI_DIGITS_LABEL_TYPE:
        scores: Dict[TI_DIGITS_LABEL_TYPE, float] = {label: model.predict(signal) for label, model in self._models.items()}
        logger.debug(f"Scores for signal is {scores}")
        sorted_scores: List[TI_DIGITS_LABEL_TYPE] = [label for label, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        logger.debug(f"Sorted labels are {sorted_scores}")
        return sorted_scores[0]

    @classmethod
    def load_from_files(cls, folder_path: str, num_of_states: int, dim_of_feature: int) -> Self:
        mc = cls()

        for label in TI_DIGITS_LABELS:
            folder_name: str = f"{label}#{num_of_states}#{dim_of_feature}"
            model_folder_path: str = os.path.join(folder_path, folder_name)
            hmm = HiddenMarkovModel.from_file(model_folder_path)
            mc._models[label] = hmm

        return mc
