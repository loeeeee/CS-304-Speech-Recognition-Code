from dataclasses import field, dataclass
from typing import Dict, List, Self, Tuple
import logging

from numpy.typing import NDArray
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SparseMatrix:
    num_of_states: int = field(default=0)
    # Internals
    _core: Dict[Tuple[int, ...], float] = field(default_factory=dict)

    def __getitem__(self, key: Tuple[int, ...]) -> float:
        assert any(key) < self.num_of_states
        assert any(key) >= 0
        if key in self._core:
            return self._core[key]
        else:
            return 0.

    def __setitem__(self, key: Tuple[int, ...], value: float) -> None:
        assert any(key) < self.num_of_states
        assert any(key) >= 0
        if key in self._core:
            logger.warning(f"{key} already in the matrix")
        self._core[key] = value

    def __str__(self) -> str:
        return f"{self._core}"

    def to_array(self) -> NDArray[np.float32]:
        ...


@dataclass
class TransitionProbabilities(SparseMatrix):

    @classmethod
    def from_num_of_states(cls, num_of_states: int) -> Self:
        transition_probabilities: List[List[float]] = []
        for current_state_index in range(num_of_states):
            inner_list: List[float] = [0. for _ in range(current_state_index)]
            inner_list.extend(\
                [1/(num_of_states - current_state_index) \
                    for _ in range(num_of_states - current_state_index)])
            transition_probabilities.append(inner_list)
        transition_probs = np.array(transition_probabilities, dtype=np.float32)
        return cls.from_transition_probability(transition_probs)

    @classmethod
    def from_transition_probability(cls, transition_probability: NDArray[np.float32]) -> Self:
        ltp = cls(transition_probability.shape[0])
        row_counter: int = -1
        column_starting_point: int = 0
        for row in transition_probability:
            row_counter += 1
            for column_index, probability in enumerate(row):
                if probability != -float("inf"):
                    ltp[(row_counter, column_starting_point + column_index)] = probability
        return ltp


@dataclass
class LogTransitionProbabilities(SparseMatrix):
    
    def append(self, ltp: Self) -> None:
        for point, value in ltp._core.items():
            new_point = tuple([i + self.num_of_states for i in point])
            self[new_point] = value

    @classmethod
    def from_transition_probability(cls, tp: TransitionProbabilities) -> Self:
        ltp = cls(tp.num_of_states)
        for point, value in tp._core.items():
            ltp[point] = np.log(value)
        return ltp
