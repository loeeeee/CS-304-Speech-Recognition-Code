from dataclasses import dataclass, field
import logging
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from tabulate import tabulate
import uniplot

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class Signal:
    # Mains
    num_of_state: int
    signal: NDArray[np.float32]
    path: NDArray[np.int8]
    # dim_of_feature: int = field(default=39)

    @property
    def order_by_state(self) -> List[NDArray|None]:
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
            
        sorted_segments: List[NDArray|None] = [segments[key] for key in sorted(segments.keys())]

        return sorted_segments
    
    @property
    def order_by_signal(self) -> List[Tuple[NDArray, int]]:
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
    def order_by_state(self) -> List[List[NDArray]]:
        signals_by_state: List[List[NDArray]] = [[] for _ in range(self.num_of_states)]
        for signal in self._signals:
            for state, signal in enumerate(signal.order_by_state):
                if not signal is None:
                    # Skip when some state has no signal
                    signals_by_state[state].append(signal)
                else:
                    logger.debug(f"state: {state} is empty when organizing signals")
        return signals_by_state

    @property
    def transition_probabilities(self) -> NDArray[np.float32]:
        transition_counts: NDArray = np.zeros((self.num_of_states, self.num_of_states), dtype=np.int32)
        for signal in self._signals:
            last_state: int = signal.path[0]
            for current_state in signal.path[1:]:
                transition_counts[last_state, current_state] += 1
                last_state = current_state
        logger.debug(f"Transition count is {transition_counts}")
        transition_probs = (transition_counts / np.sum(transition_counts, axis=1, keepdims=True)).astype(np.float32)
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
            path.append((last_state, counter))
            logger.info(f"Viterbi path: {path}")
        return
