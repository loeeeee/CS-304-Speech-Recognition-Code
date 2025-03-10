import numpy as np

from typing import List
import logging
import math
from dataclasses import dataclass, field

from .mfcc import MFCC

logger = logging.getLogger(__name__)


@dataclass
class DynamicTimeWarping:
    sequences: List[np.ndarray] # A list of words mfccs
    sample: np.ndarray
    sample_rate: int|float = field(default=16000)
    trace_back: bool = field(default=False)
    pruning: bool = field(default=True)
    pruning_factor: float = field(default=4)

    _sequences: np.ndarray = field(init=False)
    _sample: np.ndarray = field(init=False)
    _cost_matrix: np.ndarray = field(init=False)
    _path_matrix: np.ndarray = field(init=False)
    _number_of_words_in_sequences: int = field(init=False)
    _word_length_in_sequences: List[int] = field(init=False)
    _word_starting_positions: List[int] = field(init=False)
    _height: int = field(init=False)
    _length: int = field(init=False)

    def __post_init__(self):
        logger.info("Start Post Init Process")
        logger.info(f"Example sequences shape: {self.sequences[0].shape}")
        sequences_mfccs = [MFCC(word, sample_rate=self.sample_rate).feature_vector.T for word in self.sequences]
        logger.info(f"Example word length in sequences: {sequences_mfccs[0].shape[0]}")
        self._word_length_in_sequences = [word.shape[0] for word in sequences_mfccs]
        self._sequences = np.concatenate(sequences_mfccs)
        logger.info(f"Example _sequences shape: {self._sequences.shape}")
        self._sample = MFCC(self.sample, sample_rate=self.sample_rate).feature_vector.T
        self._number_of_words_in_sequences = len(self.sequences)


        self._height = self._sequences.shape[0]
        self._length = self._sample.shape[0]
        
        ## Init cost matrix
        self._cost_matrix = np.zeros((self._height+1, self._length+1))
        self._cost_matrix[1:, 0] = math.inf
        word_starting_position: int = 0
        self._cost_matrix[word_starting_position, 1:] = math.inf
        self._word_starting_positions = [0]
        for word_length in self._word_length_in_sequences[:-1]:
            word_starting_position += word_length
            self._cost_matrix[word_starting_position, 1:] = math.inf
            self._cost_matrix[word_starting_position, 0] = 0
            self._word_starting_positions.append(word_starting_position)

        logger.info(f"Length of word starting positions: {len(self._word_starting_positions)}")
        logger.info(f"Length of _number_of_words_in_sequences: {self._number_of_words_in_sequences}")

        self._path_matrix = np.zeros((self._height+1, self._length+1), dtype=int)  # 0: start, 1: up, 2: super diagonal, 3: diagonal
        logger.info("Finish Post Init Process")
        return

    def search(self):
        # Fill in the cost matrix and path matrix
        # cost_matrix_left = np.copy(self._cost_matrix)
        # cost_matrix_right = np.copy(self._cost_matrix)
        min_cost_in_column = np.full(self._length + 1, math.inf)
        # min_cost_in_column[0] = 0.0
        for j in range(1, self._length + 1):
            min_cost_in_column[j] = math.inf
            for starting_position, word_length in zip(self._word_starting_positions, self._word_length_in_sequences):
                # logger.info(f"Cost at starting position: {self._cost_matrix[starting_position, 0]} at {starting_position}")
                # logger.info(f"Cost at non-starting position: {self._cost_matrix[starting_position+1, 0]} at {starting_position+1}")
                for i in range(starting_position, starting_position + word_length + 1):
                    cost = self.euclidean_distance(self._sequences[i - 1], self._sample[j - 1])
                    insertion_cost = self._cost_matrix[i, j - 1] # Level
                    if i - 2 < starting_position:
                        shrink_cost = math.inf
                    else:
                        shrink_cost = self._cost_matrix[i - 2, j - 1] # Super Diagonal
                    match_cost = self._cost_matrix[i - 1, j - 1] # Dia

                    min_cost = min(insertion_cost, shrink_cost, match_cost)
                    current_accumulated_cost = cost + min_cost
                    
                    if self.pruning:
                        pruning_threshold = min_cost_in_column[j-1] * (1 + self.pruning_factor)
                        if current_accumulated_cost > pruning_threshold:
                            # logger.info(f"Current cost: {current_accumulated_cost}")
                            # logger.info(f"Punning happened, threshold: {pruning_threshold}")
                            self._cost_matrix[i, j] = math.inf # Prune this path
                            continue # Skip updating path matrix and min_cost_in_column for this cell
                    
                    self._cost_matrix[i, j] = current_accumulated_cost

                    if self.trace_back:
                        if min_cost == insertion_cost:
                            self._path_matrix[i, j] = 1  # Level (Insertion)
                        elif min_cost == shrink_cost:
                            self._path_matrix[i, j] = 2  # Super diagonal
                        else: # min_cost == match_cost (or could be equal to multiple, diagonal preferred if tied - standard DTW)
                            self._path_matrix[i, j] = 3  # Diagonal (Match)

                    if self._cost_matrix[i, j] != math.inf: # Only update min_cost_in_column if not pruned
                        min_cost_in_column[j] = min(min_cost_in_column[j], self._cost_matrix[i, j])

        distance_results = []
        for position, length in zip(self._word_starting_positions, self._word_length_in_sequences):
            distance_results.append(self._cost_matrix[position + length - 1, self._length])
        
        min_distance = min(distance_results)
        index = distance_results.index(min_distance)
        return index, min_distance

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))
