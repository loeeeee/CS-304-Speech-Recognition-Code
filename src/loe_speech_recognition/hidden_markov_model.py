from dataclasses import dataclass, field
import logging
from typing import List, Self, Tuple

import numpy as np
import scipy as sp

logger = logging.getLogger(__name__)

@dataclass
class SortedSignal:
    # Mains
    num_of_state: int
    signal: np.ndarray
    path: List[int]
    # dim_of_feature: int = field(default=39)

    @property
    def order_by_state(self) -> List[np.ndarray|None]:
        segments = {}
        start_index = 0
        for state_write_to in range(self.num_of_state):
            end_index = start_index
            for state_now_in in self.path[start_index:]:
                if state_now_in == state_write_to:
                    # Here we assume the state is continuous and only go up
                    end_index += 1
                else:
                    # Find the end of the sequence
                    break
            logger.debug(f"Find start index: {start_index}, end index: {end_index}")

            if start_index < end_index:
                segments[state_write_to] = self.signal[start_index: end_index]
            else:
                segments[state_write_to] = None
                logger.debug(f"Find empty state")
            start_index = end_index
            
        sorted_segments: List[np.ndarray|None] = [segments[key] for key in sorted(segments.keys())]

        return sorted_segments
    
    @property
    def order_by_signal(self) -> List[Tuple[np.ndarray, int]]:
        return [(signal, state) for signal, state in zip(self.signal, self.path)]


@dataclass
class HiddenMarkovModel:
    # Mains
    num_of_states: int
    dim_of_feature: int = field(default=39)
    label: str = field(default="")

    # Internals
    _transition_prob: np.ndarray = field(init=False)
    _means: np.ndarray = field(init=False)
    _covariances: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Init all parameters
        
        ## Set up dimension
        self._transition_prob   = np.zeros((self.num_of_states, self.num_of_states))
        self._means             = np.zeros((self.num_of_states, self.dim_of_feature))
        self._covariances       = np.zeros((self.num_of_states, self.dim_of_feature, self.dim_of_feature))

        logger.info(f"Finish initialize HMM for {str(self)}")
        return

    @classmethod
    def from_data(
        cls, 
        label: str, 
        num_of_states: int, 
        train_data: List[np.ndarray], # A list of mfcc feature vector of each signal
        dim_of_feature: int = 39,
        k_means_max_iteration: int = 10
        ) -> Self:

        hmm = cls(num_of_states, dim_of_feature=dim_of_feature, label=label)

        # Transition probabilities
        ## It should not go back
        ## | 0.333 0.333 0.333 |
        ## | 0     0.5   0.5   |
        ## | 0     0     1     |

        transition_probabilities: List[List[float]] = []
        for current_state_index in range(num_of_states):
            inner_list: List[float] = [0. for _ in range(current_state_index)]
            inner_list.extend([1/(num_of_states - current_state_index) for _ in range(num_of_states - current_state_index)])
            transition_probabilities.append(inner_list)
        ## Assign the calculation result to hmm
        hmm._transition_prob = np.array(transition_probabilities)

        # Segmental K-means
        ## Create initial state
        hmm._covariances = np.tile(np.eye(dim_of_feature), (num_of_states, 1, 1)) * 0.01  # Small diagonal covariances
        
        ## Initialize means
        means: List[float] = [0. for _ in range(num_of_states)]
        first_signal = train_data[0]
        logger.info(f"First signal has shape {first_signal.shape}")
        state_length: int = int(first_signal.shape[0] / num_of_states) # Length of the time series
        for state_index in range(num_of_states):
            start_position: int = state_index * state_length
            signal_slice = first_signal[start_position: start_position+state_length, :]
            # Mean (dim_of_feature, )
            mean = np.average(signal_slice, axis=0)
            means[state_index] = mean
        hmm._means = np.array(means)
        logger.debug("Finish compute means")
        
        for i in range(k_means_max_iteration):
            hmm.train(train_data=train_data)
            logger.info(f"Training... {i} iter")

        return hmm

    def __str__(self) -> str:
        return self.label

    def train(self, train_data) -> None:
        # 2a. Segmentation (Viterbi):
        sorted_signals: List[SortedSignal] = []
        # segments: List[List[np.ndarray]] = []  # List to store segments for each sequence
        for sequence in train_data:
            viterbi_path = self._viterbi(sequence, self.num_of_states, self._means, self._transition_prob, self._covariances)  # See function below
            sorted_signals.append(SortedSignal(self.num_of_states, sequence, viterbi_path))
            # segments.append(self._segment_data(sequence, viterbi_path)) # See function below

        # Update parameters
        ## Update means
        signal_concat_by_state: List = [[] for _ in range(self.num_of_states)]
        for sorted_signal in sorted_signals:
            logger.info(f"Next signal")
            for state, signal in enumerate(sorted_signal.order_by_state):
                logger.debug(f"state: {state}")
                if not signal is None:
                    # Skip when some state has no signal
                    logger.debug(f"signal shape: {signal.shape}")
                    signal_concat_by_state[state].append(signal)
        signal_concat_by_state = [np.concatenate(i) for i in signal_concat_by_state]
        _average = [np.average(i, axis=0) for i in signal_concat_by_state]
        logger.info(f"Average: {_average[0].shape}, {_average[1].shape}, {_average[2].shape}, {_average[3].shape}, {_average[4].shape}")
        self._means = np.array(_average) # Calculate mean for all time series, keeping feature dimension
        ## Update covariance
        for state, signals in enumerate(signal_concat_by_state):
            logger.info(f"Local Segments has shape: {signals.shape}")
            self._covariances[state] = np.cov(signals, rowvar=False) + np.eye(self.dim_of_feature) * 0.001
        ## Update transition probability
        self._transition_prob = self._estimate_transition_prob(sorted_signals, self.num_of_states)

    @staticmethod
    def _viterbi(observation_sequence: np.ndarray, num_of_states: int, means: List[float]|np.ndarray, transition_probs: np.ndarray, covariances: np.ndarray) -> List[int]:
        T: int = observation_sequence.shape[0]
        N: int = num_of_states

        log_delta = np.zeros((T, N)) # log-probabilities of the most likely path ending in state j at time t
        psi = np.zeros((T, N), dtype=int)   

        # Initialization (t=0)
        for s in range(N):
            log_delta[0, s] = sp.stats.multivariate_normal.logpdf(observation_sequence[0], means[s], covariances[s], allow_singular=False) # Assuming equal prior probabilities for starting states

        # Recursion (t=1 to T-1)
        for t in range(1, T):
            for j in range(N):
                max_log_prob = -float('inf')
                best_prev_state = 0
                for i in range(N):
                    log_prob = log_delta[t-1, i] + np.log(transition_probs[i, j])
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_state = i
                log_delta[t, j] = max_log_prob + sp.stats.multivariate_normal.logpdf(observation_sequence[t], means[j], covariances[j], allow_singular=False)
                psi[t, j] = best_prev_state

        # Termination
        best_path_prob = np.max(log_delta[T-1, :])
        last_state = np.argmax(log_delta[T-1, :])

        # Backtracking
        viterbi_path: List = [0 for _ in range(T)]
        viterbi_path[T-1] = last_state
        for t in range(T-2, -1, -1):
            viterbi_path[t] = psi[t+1, viterbi_path[t+1]]

        return viterbi_path # [np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(1), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(2), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(3), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4), np.int64(4)]

    @staticmethod
    def _estimate_transition_prob(sorted_signals: List[SortedSignal], num_of_state: int) -> np.ndarray:
        transition_counts: np.ndarray = np.zeros((num_of_state, num_of_state))
        for sorted_signal in sorted_signals:
            last_state: int = sorted_signal.order_by_signal[0][1]
            for _, current_state in sorted_signal.order_by_signal[1:]:
                transition_counts[last_state: current_state] += 1
                last_state = current_state

        transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        return transition_probs
