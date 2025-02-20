from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Literal, Self, Tuple
import logging
import random as rm

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

    def predict_continuous_controller(self, signal: NDArray) -> List[TI_DIGITS_LABEL_TYPE]:
        pred_labels = []
        # logging.getLogger().setLevel(logging.DEBUG)
        while signal.shape[0] > 40:
            logger.info(f"Signal shape: {signal.shape}")
            pred_label, signal = self.predict_continuous(signal)
            pred_labels.append(pred_label)
        return pred_labels

    def predict_phone_controller(self, signal: NDArray) -> List[TI_DIGITS_LABEL_TYPE]:
        pred_labels = self.predict_continuous_controller(signal)
        if len(pred_labels) < 4:
            while len(pred_labels) < 4:
                pred_labels.append(*rm.sample(list(TI_DIGITS_LABELS.keys()), 1))
        elif len(pred_labels) < 7:
            while len(pred_labels) < 7:
                pred_labels.append(*rm.sample(list(TI_DIGITS_LABELS.keys()), 1))
        else:
            pred_labels = rm.sample(pred_labels, 7)
        return pred_labels

    def predict_continuous(self, signal: NDArray[np.float32]) -> Tuple[TI_DIGITS_LABEL_TYPE, NDArray]:
        models_in_list: List[Tuple[TI_DIGITS_LABEL_TYPE, HiddenMarkovModel]] = [(label, model) for label, model in self._models.items()]
        best_label: TI_DIGITS_LABEL_TYPE = models_in_list[0][0]
        max_score, best_path = models_in_list[0][1].predict_with_hack(signal)
        for label, model in models_in_list[1:]:
            score, path = model.predict_with_hack(signal)
            if score > max_score and path.shape:
                best_label = label
                max_score = score
                best_path = path
        rest_of_signal: NDArray = signal[best_path.shape[0]:,:]
        return best_label, rest_of_signal

    @classmethod
    def load_from_files(cls, folder_path: str, num_of_states: int, dim_of_feature: int) -> Self:
        mc = cls()

        for label in TI_DIGITS_LABELS:
            folder_name: str = f"{label}#{num_of_states}#{dim_of_feature}"
            model_folder_path: str = os.path.join(folder_path, folder_name)
            hmm = HiddenMarkovModel.from_file(model_folder_path)
            mc._models[label] = hmm

        return mc
