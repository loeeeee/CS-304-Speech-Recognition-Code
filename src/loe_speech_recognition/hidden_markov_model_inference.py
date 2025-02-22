from dataclasses import dataclass, field
from typing import Dict, Tuple

from loe_speech_recognition.ti_digits import TI_DIGITS_LABEL_TYPE

@dataclass
class HiddenMarkovModelInference:
    num_of_states: int
    dim_of_features: int

    # Internal
    _label_locations: Dict[Tuple[int, int], TI_DIGITS_LABEL_TYPE] = field(default_factory=dict)