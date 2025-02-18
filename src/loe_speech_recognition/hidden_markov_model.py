from dataclasses import dataclass, field
import logging
from typing import Dict, List, Self, Tuple

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
                    logger.debug(f"state: {state} is empty")
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
    num_of_states: int
    dim_of_feature: int = field(default=39)
    label: str = field(default="")

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
        k_means_max_iteration: int = 10
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
        
        for i in range(k_means_max_iteration):
            hmm.train(train_data=train_data)
            logger.info(f"Training... {i} iter")

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
            except HMMTrainConverge:
                logger.info(f"🏁 Finish training at {i} iteration, after {self._initializer.init_counter} init attempt")
                break
            
            bar.update()

        return

    def train_routine_new(self, train_data) -> None:
        # Segmentation
        return


    def train_routine(self, train_data: List[np.ndarray]) -> None:
        # Segmentation
        sorted_signals: SortedSignals = SortedSignals(self.num_of_states)
        bar = tqdm(desc="Viterbi", total=len(train_data), position=0, disable=True)
        for sequence in train_data:
            viterbi_path = self._viterbi(sequence, self.num_of_states, self._means, self._transition_prob, self._covariances)  # See function below
            sorted_signals.append(Signal(self.num_of_states, sequence, viterbi_path))
            bar.update()

        sorted_signals.show_viterbi_path_histogram()

        # Update parameters
        
        ## Update means
        signals_sorted_by_state: List[List[np.ndarray]] = sorted_signals.order_by_state
        try:
            signal_concat_by_state = [np.concatenate(i) for i in signals_sorted_by_state]
        except ValueError:
            raise HMMTrainMeanFail
        ### Do the update
        new_means = [np.average(i, axis=0) for i in signal_concat_by_state]
        # logger.info(f"Average: {new_means[0].shape}, {new_means[1].shape}, {new_means[2].shape}, {new_means[3].shape}, {new_means[4].shape}")
        if np.allclose(new_means, self._means):
            logger.debug("Converges")
            raise HMMTrainConverge
        
        self._means = np.array(new_means) # Calculate mean for all time series, keeping feature dimension

        ## Update covariance
        for state, signals in enumerate(signal_concat_by_state):
            logger.info(f"Local Segments has shape: {signals.shape}")
            self._covariances[state] = np.cov(signals, rowvar=False) + np.eye(self.dim_of_feature) * 0.001
        
        ## Update transition probability
        self._transition_prob = sorted_signals.transition_probabilities.T

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
