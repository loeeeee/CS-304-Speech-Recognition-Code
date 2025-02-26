from dataclasses import dataclass, field
import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ModelBoundary:
    # Internals
    ## Boundaries: Stores word starting position besides the first and the last word
    _boundaries: List[int] = field(default_factory=list)
    _labels: List[str] = field(default_factory=list)

    # Cache
    _cache_lower_boundaries: List[int] = field(init=False)
    _cache_upper_boundaries: List[int] = field(init=False)

    # Flag
    _isFrozen: bool = field(default=False)

    @property
    def lower_boundaries(self) -> List[int]:
        """
        Get lower boundary of each word, meaning its starting state index

        Returns:
            List[int]: A list of starting state index
        """
        if not hasattr(self, "_cache_lower_boundaries"):
            new_boundaries = [0]
            new_boundaries.extend(self._boundaries[:-1])
            new_boundaries = sorted(new_boundaries)
            self._cache_lower_boundaries = new_boundaries
            self._isFrozen = True
        return self._cache_lower_boundaries

    @property
    def upper_boundaries(self) -> List[int]:
        """
        Get upper boundary of each word, meaning its ending state index

        Returns:
            List[int]: A list of ending state index
        """
        if not hasattr(self, "_cache_upper_boundaries"):
            new_boundaries = self._boundaries.copy()
            new_boundaries = [i - 1 for i in new_boundaries]
            new_boundaries = sorted(new_boundaries)
            self._cache_upper_boundaries = new_boundaries
            self._isFrozen = True
        return self._cache_upper_boundaries

    @property
    def num_of_words(self) -> int:
        return len(self._boundaries)

    def append(self, num_of_states: int) -> None:
        """
        Note the number of states current word has

        Args:
            num_of_states (int): Number of states
        """
        if self._isFrozen:
            logger.error("Adding data after cache is created")
            raise Exception
        try:
            self._boundaries.append(
                self._boundaries[-1] + num_of_states
            )
        except IndexError:
            self._boundaries.append(
                num_of_states
            )
            logger.info(f"Init boundary internal array")
        return

    def find_lower_boundary(self, state: int) -> int:
        for lower_boundary in reversed(self.lower_boundaries):
            if state >= lower_boundary:
                logger.debug(f"Find lower boundary {lower_boundary}")
                return lower_boundary
            else:
                continue
        logger.error(f"Failed to find lower boundary for state {state}")
        raise Exception

    def find_upper_boundary(self, state: int) -> int:
        for upper_boundaries in self.upper_boundaries:
            if state <= upper_boundaries:
                logger.debug(f"Find upper boundary {upper_boundaries}")
                return upper_boundaries
            else:
                continue
        logger.error(f"Failed to find upper boundary for state {state}")
        raise Exception

    def add_model_labels(self, model_labels: List[str]) -> None:
        assert len(model_labels) == self.num_of_words
        self._labels = model_labels
        return

    def get_labels(self, path: NDArray[np.int8], skip_silence: bool=True) -> List[str]:
        path_list: List[int] = path.tolist()
        # Parse the path
        ## When big jump happens
        last_point: int = path_list[0]
        simplified_compressed_path: List[int] = [last_point]
        for current_point in path_list[1:]:
            if current_point != last_point:
                simplified_compressed_path.append(current_point)
                last_point = current_point
        logger.debug(f"Viterbi path: {simplified_compressed_path}")

        labels: List[str] = []
        first_point = simplified_compressed_path[0]
        lower_boundary = self.find_lower_boundary(first_point)
        upper_boundary = self.find_upper_boundary(first_point)
        self.append_to_labels(first_point, skip_silence, labels)
        for index, current_point in enumerate(simplified_compressed_path[1:], start=1):
            if current_point < lower_boundary or current_point > upper_boundary:
                # Discover new word, straightforward scenario
                lower_boundary = self.find_lower_boundary(current_point)
                upper_boundary = self.find_upper_boundary(current_point)
                # Note the word
                ## Deal with silence word
                self.append_to_labels(current_point, skip_silence, labels)
            else:
                last_point = simplified_compressed_path[index - 1]
                if last_point == upper_boundary and current_point == lower_boundary:
                    # Discover new word, when previous word is repeated
                    self.append_to_labels(current_point, skip_silence, labels)

        logger.debug(f"Find labels {labels}")
        return labels

    def append_to_labels(self, state: int, skip_silence: bool, labels: List[str]) -> None:
        current_label = self.get_label(state)
        if current_label == "S" and skip_silence:
            # Skip "S" if not include_silence
            pass
        else:
            labels.append(current_label) # Modify the labels

    def get_label(self, state: int) -> str:
        """
        Given a state, find the corresponding label

        Args:
            state (int): A state

        Returns:
            str: The label
        """
        lower_boundary = self.find_lower_boundary(state)
        label_index: int = self.lower_boundaries.index(lower_boundary)
        label: str = self._labels[label_index]
        logger.debug(f"Get label index {label_index} for state {state}, corresponding to label {label}")
        return label

    def get_state_range(self, label: str) -> Tuple[int, int]:
        """
        Get the range of state corresponding to given label

        Args:
            label (str): The interested label

        Returns:
            List[int]: A list of corresponding states
        """
        label_index = self._labels.index(label)
        if label_index == 0:
            return (0, self._boundaries[label_index])
        else:
            return (self._boundaries[label_index - 1], self._boundaries[label_index])
