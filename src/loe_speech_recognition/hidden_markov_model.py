from dataclasses import dataclass, field
import logging
import math
from typing import Dict, List, Self, Tuple
import os
import time
import functools

import numpy as np
import scipy as sp
from tqdm import tqdm
from tabulate import tabulate
import uniplot

logger = logging.getLogger(__name__)


@dataclass
class Signal:
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
            # logger.debug(f"Find start index: {start_index}, end index: {end_index}")

            if start_index < end_index:
                segments[state_write_to] = self.signal[start_index: end_index]
            else:
                segments[state_write_to] = None
                # logger.debug(f"Find empty state")
            start_index = end_index
            
        sorted_segments: List[np.ndarray|None] = [segments[key] for key in sorted(segments.keys())]

        return sorted_segments
    
    @property
    def order_by_signal(self) -> List[Tuple[np.ndarray, int]]:
        return [(signal, state) for signal, state in zip(self.signal, self.path)]

@dataclass
class SortedSignals:
    # Mains
    num_of_states: int

    # Internals
    _signals: List[Signal] = field(init=False)

    def __post_init__(self) -> None:
        self._signals = []
        return

    def append(self, signal: Signal) -> None:
        self._signals.append(signal)

    @property
    def order_by_state(self) -> List[List[np.ndarray]]:
        signals_by_state: List[List[np.ndarray]] = [[] for _ in range(self.num_of_states)]
        for signal in self._signals:
            for state, signal in enumerate(signal.order_by_state):
                if not signal is None:
                    # Skip when some state has no signal
                    signals_by_state[state].append(signal)
                else:
                    logger.debug(f"state: {state} is empty when organizing signals")
        return signals_by_state

    @property
    def transition_probabilities(self) -> np.ndarray:
        transition_counts: np.ndarray = np.zeros((self.num_of_states, self.num_of_states))
        for signal in self._signals:
            last_state: int = signal.path[0]
            for current_state in signal.path[1:]:
                transition_counts[last_state: current_state] += 1
                last_state = current_state
        transition_probs = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
        return transition_probs

    def show_viterbi_path_table(self) -> None:
        counter: Dict[int, int] = {}
        for signal in self._signals:
            for i in signal.path:
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1
        
        counter_tab: List[Tuple[int, int]] = [(state, count) for state, count in counter.items()]
        table = tabulate(counter_tab, ["State", "Count"], tablefmt="grid")
        descriptions: List[str] = table.split("\n")
        for i in descriptions:
            logger.debug(i)
        return
    
    def show_viterbi_path_histogram(self) -> None:
        counter: List[int] = []
        for signal in self._signals:
            counter.extend(signal.path)
        uniplot.histogram(counter, bins=10, bins_min=0, bins_max=self.num_of_states)

    def show_viterbi_path_str(self) -> None:
        # paths: List[List[Tuple[int, int]]] = []
        for signal in self._signals:
            path: List[Tuple[int, int]] = []
            counter: int = 1
            last_state: int = int(signal.path[0])
            for i in signal.path[1:]:
                current_state: int = i
                if current_state != last_state:
                    path.append((last_state, counter))
                    last_state = int(current_state)
                    counter = 1
                else:
                    counter += 1
            # paths.append(path)
            logger.debug(f"Viterbi path: {path}")
        return


class HMMTrainMeanFail(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        logger.warning(f"Cannot use all the state for the HMM")


class HMMTrainConverge(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        logger.warning("Successfully train the HMM model")


@dataclass
class HiddenMarkovModelInitializer:
    # Mains
    num_of_states: int
    train_data: List[np.ndarray]
    dim_of_feature: int = field(default=39)

    # Internals
    _init_seed: int = field(default=0)

    def __post_init__(self) -> None:
        ## Set up dimension
        logger.info(f"Created HMM initializer")
        return

    def init_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Transition Probabilities
        transition_probabilities: List[List[float]] = []
        for current_state_index in range(self.num_of_states):
            inner_list: List[float] = [0. for _ in range(current_state_index)]
            inner_list.extend([1/(self.num_of_states - current_state_index) for _ in range(self.num_of_states - current_state_index)])
            transition_probabilities.append(inner_list)
        transition_probs = np.array(transition_probabilities)
        logger.debug("Finish compute transition probabilities")

        # Means
        means: List[float] = [0. for _ in range(self.num_of_states)]
        first_signal = self.train_data[self._init_seed]
        logger.info(f"First signal has shape {first_signal.shape}")
        state_length: int = int(first_signal.shape[0] / self.num_of_states) # Length of the time series
        for state_index in range(self.num_of_states):
            start_position: int = state_index * state_length
            signal_slice = first_signal[start_position: start_position+state_length, :]
            # Mean (dim_of_feature, )
            mean = np.average(signal_slice, axis=0)
            means[state_index] = mean
        logger.debug("Finish compute means")

        # Covariance
        covariance = np.tile(np.eye(self.dim_of_feature), (self.num_of_states, 1, 1)) * 0.01

        # Init seed
        self._init_seed += 1
        return transition_probs, np.array(means), covariance

    @property
    def init_counter(self) -> int:
        return self._init_seed


@dataclass
class HiddenMarkovModel:
    # Mains
    num_of_states: int = field(default=5)
    dim_of_feature: int = field(default=39)
    label: str = field(default_factory=functools.partial(time.strftime, "%Y%m%d-%H%M%S")) # Default label to timestamp

    # Settings
    isTqdm: bool = field(default=True)

    # Internals
    _transition_prob: np.ndarray = field(init=False)
    _means: np.ndarray = field(init=False)
    _covariances: np.ndarray = field(init=False)
    _initializer: HiddenMarkovModelInitializer = field(init=False)

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
        k_means_max_iteration: int = 100
        ) -> Self:

        hmm = cls(num_of_states, dim_of_feature=dim_of_feature, label=label)

        # Transition probabilities
        ## It should not go back
        ## | 0.333 0.333 0.333 |
        ## | 0     0.5   0.5   |
        ## | 0     0     1     |

        hmm._initializer = HiddenMarkovModelInitializer(
            num_of_states=num_of_states,
            train_data=train_data,
            dim_of_feature=dim_of_feature
        )
        
        hmm.train(train_data=train_data, k_means_max_iteration=k_means_max_iteration)
        return hmm

    @classmethod
    def from_file(cls, folder_path: str) -> Self:
        hmm = cls()
        
        logger.info(f"Loading files from {folder_path}")
        model_label: str = folder_path.split("/")[-1]
        logger.info(f"Find model label {model_label}")
        hmm.label = model_label

        # Save transition
        trans_probs_file_name: str = os.path.join(folder_path, "trans_probs.npy")
        hmm._transition_prob = np.load(trans_probs_file_name)

        # Save means
        means_file_name: str = os.path.join(folder_path, "means.npy")
        hmm._means = np.load(means_file_name)

        # Save covariances
        covariances_file_name: str = os.path.join(folder_path, "covariances.npy")
        hmm._covariances = np.load(covariances_file_name)

        logger.info(f"Finish loading all files for {model_label} model")
        return hmm

    def __str__(self) -> str:
        return self.label

    def train(self, train_data, k_means_max_iteration: int = 100) -> None:
        force_init: bool = True
        bar = tqdm(desc="Main Train", total=k_means_max_iteration, position=1, disable=not self.isTqdm)
        for i in range(k_means_max_iteration):
            if force_init:
                self._transition_prob, self._means, self._covariances = self._initializer.init_values()
                logger.debug("Init the three parameters")
                force_init = False

            try:
                self.train_routine(train_data=train_data)
            except HMMTrainMeanFail:
                force_init = True
                logger.warning(f"Reinit parameters next loop, {i+1}")
                raise # for debug
            except HMMTrainConverge:
                logger.info(f"🏁 Finish training at {i} iteration, after {self._initializer.init_counter} init attempt")
                break
            
            bar.update()
        bar.close()

        return
    
    def predict(self, signal: np.ndarray) -> float:
        ...

    def save(self, folder_path: str = "./cache") -> None:
        model_folder: str = os.path.join(folder_path, self.label)
        logger.info(f"Saving files to {model_folder}")
        os.mkdir(model_folder)

        # Save transition
        trans_probs_file_name: str = os.path.join(model_folder, "trans_probs.npy")
        np.save(trans_probs_file_name, self._transition_prob)

        # Save means
        means_file_name: str = os.path.join(model_folder, "means.npy")
        np.save(means_file_name, self._means)

        # Save covariances
        covariances_file_name: str = os.path.join(model_folder, "covariances.npy")
        np.save(covariances_file_name, self._covariances)

        logger.info(f"Finish saving all files for {self.label} model")
        return

    def train_routine(self, train_data: List[np.ndarray]) -> None:
        # Segmentation
        logger.debug(f"Calculating Viterbi Path")
        sorted_signals: SortedSignals = SortedSignals(self.num_of_states)
        bar = tqdm(desc="Viterbi", total=len(train_data), position=0, disable=True)
        for sequence in train_data:
            viterbi_path, best_score = self._viterbi(sequence, self.num_of_states, self._means, self._transition_prob, self._covariances)
            sorted_signals.append(Signal(self.num_of_states, sequence, viterbi_path))
            bar.update()
        bar.close()
        
        sorted_signals.show_viterbi_path_str()

        # Update parameters
        
        ## Update means
        logger.debug(f"Calculating new means")
        signals_sorted_by_state: List[List[np.ndarray]] = sorted_signals.order_by_state
        try:
            signal_concat_by_state = [np.concatenate(i) for i in signals_sorted_by_state]
        except ValueError:
            raise HMMTrainMeanFail
        ### Do the update
        new_means = [np.average(i, axis=0) for i in signal_concat_by_state]
        logger.debug(f"Means is updated to {new_means}")
        if np.allclose(new_means, self._means):
            logger.debug("Converges")
            raise HMMTrainConverge
        
        self._means = np.array(new_means) # Calculate mean for all time series, keeping feature dimension

        ## Update covariance
        logger.debug(f"Calculating new covariance")
        for state, signals in enumerate(signal_concat_by_state):
            logger.debug(f"State signal has shape: {signals.shape}")
            self._covariances[state] = np.cov(signals, rowvar=False) + np.eye(self.dim_of_feature) * 0.001
        
        ## Update transition probability
        logger.debug(f"Calculating new transition prob")
        self._transition_prob = sorted_signals.transition_probabilities

    @staticmethod
    def _viterbi(observation_sequence: np.ndarray, num_of_states: int, means: List[float]|np.ndarray, transition_probs: np.ndarray, covariances: np.ndarray) -> Tuple[List[int], float]:
        sequence_length: int = observation_sequence.shape[0]
        # log_transition_probs: np.ndarray = np.log(transition_probs)
        likelihoods = np.full((sequence_length, num_of_states), -math.inf) # can shrink the size
        tracer: np.ndarray = np.full((sequence_length, num_of_states), -math.inf, dtype=np.int16)

        likelihoods[0, 0] = sp.stats.multivariate_normal.logpdf(\
                observation_sequence[0], means[0], covariances[0], \
                    allow_singular=False)\
                        + np.log(transition_probs[0, 0])
        logger.debug(f"Initial log likelihood is {likelihoods[0, 0]}")
        
        def get_likelihood(time: int, new_state: int, old_state: int) -> float:
            result_p1 = sp.stats.multivariate_normal.logpdf(\
                observation_sequence[time], means[new_state], covariances[new_state], \
                    allow_singular=False)
            result_p2 = np.log(transition_probs[old_state, new_state])
            result_p3 = likelihoods[time - 1, old_state]
            result = result_p1 + result_p2 + result_p3
            # logger.debug(f"At time {time}, trans from {old_state} to {new_state}, likelihood is {result_p1}, {result_p2}")
            return result

        
        for t in range(1, sequence_length):
            for state in range(num_of_states):
                # Find best likelihood
                current_max_likelihood = [-math.inf for _ in range(num_of_states)]
                for old_state in range(max(state-2, 0), state+1):
                    current_max_likelihood[old_state] = get_likelihood(t, state, old_state)
                    # logger.debug(f"Transiting from {old_state} to {state} has the likelihood of {total_likelihood}")
                max_value: float = max(current_max_likelihood)
                max_index: int = current_max_likelihood.index(max_value)
                logger.debug(f"The transition to {state} has the max likelihood of {max_value} from state {max_index}, whose likelihood is {likelihoods[t-1, max_index]}")
            
            # for state in range(num_of_states):
                # Update likelihoods
                likelihoods[t, state] = max_value
                # Note the best path for each
                tracer[t, state] = max_index

            # logger.debug(f"At time {t}, maximum likelihood becomes {np.max(likelihoods[t])}, at state {np.argmax(likelihoods[t])}")
            logger.debug(f"At time {t}, likelihoods become {likelihoods[t,:]}")

        score: float = likelihoods[-1, -1]

        # Find the path
        prev_state: int = tracer[-1, -1]
        path: List[int] = [num_of_states]
        
        for t in range(sequence_length - 2, 0, -1):
            logger.debug(f"The previous state at {t} is {prev_state}")
            path.append(prev_state)
            prev_state = tracer[t, prev_state]
        path = list(reversed(path))
        logger.debug(f"Finish viterbi, the score is {score}")
        return path, score
