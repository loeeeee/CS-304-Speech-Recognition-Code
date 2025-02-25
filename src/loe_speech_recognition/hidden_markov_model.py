from dataclasses import dataclass, field
import functools
import os
from typing import Dict, Generic, List, Self, Tuple, no_type_check
import logging
import concurrent.futures
import pickle

from numpy.typing import NDArray
import numpy as np
import scipy as sp
from tqdm import tqdm

from .ti_digits import TI_DIGITS_LABEL_TYPE
from .signal import Signal, SortedSignals

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
        for lower_boundary in self.lower_boundaries:
            if state >= lower_boundary:
                logger.debug(f"Find lower boundary {lower_boundary}")
                return lower_boundary
            else:
                continue
        logger.error(f"Failed to find lower boundary for state {state}")
        raise Exception

    def find_upper_boundary(self, state: int) -> int:
        for upper_boundaries in reversed(self.upper_boundaries):
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

    def get_labels(self, path: NDArray[np.int8]) -> List[str]:
        # path_list: List[int] = path.tolist()
        # Parse the path
        ## When big jump happens
        compressed_path: List[Tuple[int, int]] = []
        counter: int = 1
        last_state: int = int(path[0])
        for i in path[1:]:
            current_state: int = i
            if current_state != last_state:
                compressed_path.append((last_state, counter))
                last_state = int(current_state)
                counter = 1
            else:
                counter += 1
        compressed_path.append((last_state, counter))
        logger.info(f"Viterbi path: {compressed_path}") # [(0, 2), (1, 4), (2, 9), ...]

        labels: List[str] = []
        simplified_compressed_path = [i[0] for i in compressed_path]
        first_point = simplified_compressed_path[0]
        lower_boundary = self.find_lower_boundary(first_point)
        upper_boundary = self.find_upper_boundary(first_point)
        labels.append(self.get_label(first_point))
        for index, current_point in enumerate(simplified_compressed_path[1:], start=1):
            if current_point < lower_boundary or current_point > upper_boundary:
                # Discover new word, straightforward scenario
                labels.append(self.get_label(current_point))
                lower_boundary = self.find_lower_boundary(current_point)
                upper_boundary = self.find_upper_boundary(current_point)
            else:
                last_point = simplified_compressed_path[index - 1]
                if last_point == upper_boundary and current_point == lower_boundary:
                    # Discover new word, when previous word is repeated
                    labels.append(self.get_label(current_point))

        logger.info(f"Find labels {labels}")
        return labels

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
        logger.debug(f"Get label index {label_index}, corresponding to label {label}")
        return label


@dataclass
class SparseMatrix:
    # Internals
    _core: Dict[Tuple[int, ...], float] = field(default_factory=dict)

    def __getitem__(self, key: Tuple[int, ...]) -> float:
        if key in self._core:
            return self._core[key]
        else:
            return 0.

    def __setitem__(self, key: Tuple[int, ...], value: float) -> None:
        if key in self._core:
            logger.warning(f"{key} already in the matrix")
        self._core[key] = value


@dataclass
class MultivariateNormal:
    dim_of_features: int = field(init=False)

    # Internal
    _core: sp.stats._multivariate.multivariate_normal_frozen = field(init=False)

    @classmethod
    def from_means_covariances(
        cls, 
        mean: NDArray[np.float32], 
        covariance: NDArray[np.float32]
        ) -> Self:
        
        mn = cls()

        mn._core = sp.stats.multivariate_normal(
            mean=mean, 
            cov=covariance, 
            allow_singular=False
            )

        mn.dim_of_features = mean.shape[0]

        return mn
    
    def log_pdf(self, x: NDArray) -> float:
        assert x.shape[0] == self.dim_of_features
        return self._core.logpdf(x).astype(np.float32)


@dataclass
class HiddenMarkovModel:
    # Mains
    label: str

    # Settings
    isMultiProcessing: bool = field(default=True)
    isTqdm: bool = field(default=True)

    # Internals
    _multivariate_normals: List[MultivariateNormal] = field(default_factory=list)
    _log_transition_probs: NDArray[np.float32] = field(init=False)

    def __str__(self) -> str:
        return self.label

    @property
    def num_of_states(self) -> int:
        return len(self._multivariate_normals)

    @property
    def dim_of_features(self) -> int:
        return self._multivariate_normals[0].dim_of_features

    def predict(self, signal: NDArray[np.float32]) -> Tuple[float, NDArray[np.int8]]:
        assert len(self._multivariate_normals) > 0
        assert self.dim_of_features == signal.shape[1]
        return self._viterbi(signal)

    def _viterbi(self, observation_sequence: NDArray[np.float32]) -> Tuple[float, NDArray[np.int8]]:
        initial_likelihood: NDArray[np.float32] = np.full((self.num_of_states), -float("inf"), dtype=np.float32)
        initial_likelihood[0] = self._multivariate_normals[0].log_pdf(observation_sequence[0]) \
                    + self._log_transition_probs[0, 0]
        logger.debug(f"Initial log likelihood is {initial_likelihood[0]}")
        score, path = self._viterbi_static(
            observation_sequence=observation_sequence,
            log_transition_probabilities=self._log_transition_probs,
            multivariate_normals=self._multivariate_normals,
            initial_likelihood=initial_likelihood,
            )
        return score, path

    def save(self, parent_folder_path: str = "./cache") -> None:
        model_folder: str = os.path.join(
            parent_folder_path, 
            f"{self.label}")
        logger.info(f"Saving files to {model_folder}")
        if not os.path.isdir(model_folder):
            logger.info(f"Folder {model_folder} do not exists, creating the folder")
            os.makedirs(model_folder)

        # Save transition
        log_trans_probs_file_path: str = os.path.join(model_folder, "log_trans_probs.npy")
        np.save(log_trans_probs_file_path, self._log_transition_probs)

        # Save multivariate
        multivariate_normals_file_path: str = os.path.join(model_folder, "multivariate_normals.pickle")
        with open(multivariate_normals_file_path, "wb") as f:
            logger.info(f"Saving {len(self._multivariate_normals)} multivariate normals")
            pickle.dump(self._multivariate_normals, f, pickle.HIGHEST_PROTOCOL)

        logger.info(f"Finish saving all files for {self.label} model")
        return

    @classmethod
    def from_folder(cls, model_folder_path: str) -> Self:

        logger.info(f"Loading from {model_folder_path}")
        if not os.path.isdir(model_folder_path):
            logger.info(f"Folder {model_folder_path} do not exists")
            raise FileNotFoundError
        
        # Find basics
        label = cls._model_folder_name_parser(model_folder_path)
        model = cls(label)

        # Create model

        # Load transition
        log_trans_probs_file_path: str = os.path.join(model_folder_path, "log_trans_probs.npy")
        model._log_transition_probs = np.load(log_trans_probs_file_path)

        # Load multivariate
        multivariate_normals_file_path: str = os.path.join(model_folder_path, "multivariate_normals.pickle")
        with open(multivariate_normals_file_path, "rb") as f:
            model._multivariate_normals = pickle.load(f)

        logger.info(f"Finish loading all files for {str(model)} model")
        return model

    @staticmethod
    def _model_folder_name_parser(folder_path: str) -> str:
        """
        Parse folder name of each model

        Args:
            folder_path (str): Full path of the folder

        Returns:
            str: Label
        """
        label: str = folder_path.split("/")[-1]
        logger.info(f"Folder name parsed, {label}")

        return str(label)

    @classmethod
    def _viterbi_static(
        cls,
        observation_sequence: NDArray[np.float32], 
        log_transition_probabilities: NDArray[np.float32], 
        multivariate_normals: List[MultivariateNormal],
        initial_likelihood: NDArray[np.float32]
        ) -> Tuple[float, NDArray[np.int8]]:
        
        num_of_states: int = len(multivariate_normals)
        sequence_length: int = observation_sequence.shape[0]
        likelihoods_left = initial_likelihood

        # Init array
        likelihoods_right: NDArray[np.float32] = np.full((num_of_states), -float("inf"), dtype=np.float32)
        tracer: NDArray[np.int8] = np.zeros((sequence_length, num_of_states), dtype=np.int8) - 1 # RuntimeWarning: invalid value encountered in cast if full -math.inf

        for time in range(1, sequence_length):
            for new_state in range(num_of_states): # Multi processing able
                # Find best likelihood
                current_max_likelihood: NDArray[np.float32] = np.full((num_of_states, ), -float("inf"))
                for old_state in range(max(new_state-2, 0), new_state+1):
                    current_max_likelihood[old_state] = \
                            + log_transition_probabilities[old_state, new_state]\
                                + likelihoods_left[old_state]
                max_value: np.float32 = np.max(current_max_likelihood)
                max_index: np.intp = np.argmax(current_max_likelihood)

                # Add multivariate normal here so we save compute
                likelihoods_right[new_state] = max_value + multivariate_normals[new_state].log_pdf(observation_sequence[time])

                # Note the best path for each
                tracer[time, new_state] = max_index
            # Move array for next iteration
            likelihoods_left = likelihoods_right
            likelihoods_right = np.full((num_of_states), -float("inf"), dtype=np.float32)

        # score: float = likelihoods[-1, -1]
        score: float = likelihoods_left[-1]

        # Find the path
        prev_state: int = tracer[-1, -1]
        path: NDArray = np.zeros((sequence_length, ), dtype=np.int8)
        path[-1] = prev_state
        
        for time in range(sequence_length - 2, 0, -1):
            path[time] = prev_state
            prev_state = tracer[time, prev_state]
        return score, path


@dataclass
class HiddenMarkovModelTrainable(HiddenMarkovModel):
    # Flags
    class HMMTrainMeanFail(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)
            logger.warning(f"Cannot use all the state for the HMM")


    class HMMTrainConverge(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)
            logger.info("Successfully train the HMM model")

    _means: NDArray[np.float32] = field(init=False)
    _covariances: NDArray[np.float32] = field(init=False)
    _transition_probs: NDArray[np.float32] = field(init=False)

    @property
    def num_of_states(self) -> int:
        return self._means.shape[0]

    @classmethod
    def from_data(
        cls, 
        label: str,
        mfccs: List[NDArray[np.float32]], 
        num_of_states: int = 5, 
        max_iterations: int = 100,
        isMultiProcessingTraining: bool = True,
        isTqdm: bool = True,
        ) -> Self:
        # dim_of_features: int = mfccs[0].shape[1]

        model = cls(label, isMultiProcessing=isMultiProcessingTraining, isTqdm=isTqdm)

        # Init model weights
        model._means, model._covariances, model._transition_probs = \
            model._init_parameters(mfccs[0], num_of_states)
        ## Init inference
        model._update_inference_weights()

        # Add Bar
        bar = tqdm(
            desc=f"Train {model.label} model", 
            total=max_iterations, 
            position=1, 
            disable=not model.isTqdm
            )

        # Train model iteratively
        for it in range(max_iterations):
            logger.info(f"Training model {str(model)} in {it} iteration")
            try:
                model._train(mfccs)
                model._update_inference_weights()
            except cls.HMMTrainMeanFail:
                logger.error(f"Failed to train model")
                raise
            except cls.HMMTrainConverge:
                logger.info(f"Finish training model {str(model)} after {it} iterations")
                logger.info(f"trans: {model._transition_probs}")
                break
            bar.update()
        bar.close()

        # Calculate the inference necessary things
        logger.info("Calculate the inference necessary things")
        model._update_inference_weights() # This is needed because when coverage, the update would be skipped
        logger.info("Finish calculating")
        return model

    def _update_inference_weights(self) -> None:
        logger.info("Calculate the inference necessary things")
        self._log_transition_probs = np.log(self._transition_probs)
        self._multivariate_normals = self.get_multivariate_normals(
            self._means,
            self._covariances,
        )
        return

    def _train(self, mfccs: List[NDArray[np.float32]]) -> None:
        # Segmentation
        logger.debug(f"Calculating Viterbi Path")
        sorted_signals: SortedSignals = SortedSignals(self.num_of_states)

        bar = tqdm(desc="Viterbi", total=len(mfccs), position=0, disable=True)
        if self.isMultiProcessing:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for sequence, (best_score, viterbi_path) in \
                    zip(mfccs, executor.map(self._viterbi, mfccs)):
                    sorted_signals.append(Signal(self.num_of_states, sequence, viterbi_path))
                    bar.update()
        else:
            for sequence in mfccs:
                best_score, viterbi_path = self._viterbi(sequence)
                sorted_signals.append(Signal(self.num_of_states, sequence, viterbi_path))

        bar.close()

        # sorted_signals.show_viterbi_path_str()
        # sorted_signals.show_viterbi_path_histogram()

        # Update parameters
        
        ## Update means
        logger.debug(f"Calculating new means")
        signals_sorted_by_state: List[List[NDArray]] = sorted_signals.order_by_state
        try:
            signal_concat_by_state = [np.concatenate(i) for i in signals_sorted_by_state]
        except ValueError:
            raise self.HMMTrainMeanFail
        ### Do the update
        new_means = [np.average(i, axis=0) for i in signal_concat_by_state]
        logger.debug(f"Means is updated to {new_means}")
        if np.allclose(new_means, self._means):
            logger.debug("Converges")
            raise self.HMMTrainConverge
        
        self._means = np.array(new_means, dtype=np.float32) # Calculate mean for all time series, keeping feature dimension

        ## Update covariance
        logger.debug(f"Calculating new covariance")
        for state, signals in enumerate(signal_concat_by_state):
            logger.debug(f"State signal has shape: {signals.shape}")
            self._covariances[state] = (np.cov(signals, rowvar=False) \
                + np.eye(self._covariances.shape[1]) * 0.001)\
                    .astype(np.float32)
        
        ## Update transition probability
        logger.debug(f"Calculating new transition probabilities")
        self._transition_probs = sorted_signals.transition_probabilities
        logger.debug(f"Trans probabilities is {self._transition_probs}")

        return

    @staticmethod
    def _init_parameters(
        sample_signal: NDArray[np.float32], 
        num_of_states: int
        ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        dim_of_features = sample_signal.shape[1]
        # Transition Probabilities
        transition_probabilities: List[List[float]] = []
        for current_state_index in range(num_of_states):
            inner_list: List[float] = [0. for _ in range(current_state_index)]
            inner_list.extend(\
                [1/(num_of_states - current_state_index) \
                    for _ in range(num_of_states - current_state_index)])
            transition_probabilities.append(inner_list)
        transition_probs = np.array(transition_probabilities, dtype=np.float32)
        logger.debug("Finish compute transition probabilities")

        # Means
        means: List[float] = [0. for _ in range(num_of_states)]
        logger.info(f"First mfcc signal has shape {sample_signal.shape}")
        state_length: int = int(sample_signal.shape[0] / num_of_states) # Length of the time series
        for state_index in range(num_of_states):
            start_position: int = state_index * state_length
            signal_slice = sample_signal[start_position: start_position+state_length, :]
            # Mean (dim_of_features, )
            mean = np.average(signal_slice, axis=0)
            means[state_index] = mean
        logger.debug("Finish compute means")

        # Covariance
        covariance = (np.tile(np.eye(dim_of_features), (num_of_states, 1, 1)) * 0.01).astype(np.float32)

        return np.array(means, dtype=np.float32), covariance, transition_probs

    @staticmethod
    def get_multivariate_normals(
        means: NDArray[np.float32], 
        covariances: NDArray[np.float32]
        ) -> List[MultivariateNormal]:
        """
        Generate number of state of number of multivariate normals,
        given means and covariances

        Args:
            means (NDArray[np.float32]): An array with shape (num_of_states, dim_of_features)
            covariances (NDArray[np.float32]): An array with shape (num_of_states, dim_of_features, dim_of_features)

        Returns:
            List[MultivariateNormal]: Num_of_states of MN in a list
        """
        result = [MultivariateNormal.from_means_covariances(mean=mean, covariance=covariance) \
            for mean, covariance in zip(means, covariances)]
        logger.info(f"Getting {len(result)} multivariate normals")
        return result


@dataclass
class HiddenMarkovModelInference:
    
    _multivariate_normals: List[MultivariateNormal] = field(default_factory=list)
    _log_transition_probs: SparseMatrix = field(default_factory=SparseMatrix)
    _model_boundaries: ModelBoundary = field(init=False)

    @classmethod
    def from_folder(cls, folder_path: str, models_to_load: List[str]) -> Self:
        # Create new object
        hmm_inference = cls()

        log_transition_probabilities: List[NDArray[np.float32]] = []
        multivariate_normals: List[MultivariateNormal] = []
        model_labels: List[str] = []
        model_boundaries: ModelBoundary = ModelBoundary()
        # Walk the directory
        for model_folder_name in os.listdir(folder_path):
            model_folder_path = os.path.join(folder_path, model_folder_name)
            label = HiddenMarkovModel._model_folder_name_parser(model_folder_path)
            if not (label in models_to_load): # Bug is here
                logger.info(f"Skipping {model_folder_name}, because it is not models to load")
                continue

            # Load the model
            # Load transition
            log_trans_probs_file_path: str = os.path.join(model_folder_path, "log_trans_probs.npy")
            log_transition_probability = np.load(log_trans_probs_file_path)

            # Load multivariate
            multivariate_normals_file_path: str = os.path.join(model_folder_path, "multivariate_normals.pickle")
            with open(multivariate_normals_file_path, "rb") as f:
                model_multivariate_normals = pickle.load(f)
            
            # Put into structure
            log_transition_probabilities.append(log_transition_probability)
            multivariate_normals.extend(model_multivariate_normals)
            model_boundaries.append(len(model_multivariate_normals))
            model_labels.append(label)

            logger.info(f"Finish loading all files for {str(label)} model")

        row_counter: int = -1
        for log_transition_probability in log_transition_probabilities:
            column_starting_point: int = row_counter + 1
            for row in log_transition_probability:
                row_counter += 1
                for column_index, probability in enumerate(row):
                    if probability != -float("inf"):
                        hmm_inference._log_transition_probs\
                            [(row_counter, column_starting_point + column_index)] = probability

        hmm_inference._multivariate_normals = multivariate_normals
        logger.info(f"Loading {len(hmm_inference._multivariate_normals)} of multivariate normals for inference")
        # Model boundaries
        model_boundaries.add_model_labels(model_labels)
        hmm_inference._model_boundaries = model_boundaries
        
        return hmm_inference

    def predict(self, signal: NDArray[np.float32]) -> str:
        score, path = self._viterbi(observation_sequence=signal)
        labels = self._model_boundaries.get_labels(path)
        return "".join(labels)

    def _viterbi(self, observation_sequence: NDArray[np.float32]) -> Tuple[float, NDArray[np.int8]]:
        initial_likelihood: NDArray[np.float32] = np.full((len(self._multivariate_normals)), -float("inf"), dtype=np.float32)
        for lower_boundary in self._model_boundaries.lower_boundaries:
            initial_likelihood[lower_boundary] = self._multivariate_normals[lower_boundary].log_pdf(observation_sequence[0]) \
                        + self._log_transition_probs[lower_boundary, lower_boundary]
        logger.debug(f"One of initial log likelihood is {initial_likelihood[0]}")

        logger.debug(f"Start finding best match using viterbi static")
        score, path = self._viterbi_static(
            observation_sequence=observation_sequence,
            log_transition_probabilities=self._log_transition_probs,
            multivariate_normals=self._multivariate_normals,
            initial_likelihood=initial_likelihood,
            model_boundaries=self._model_boundaries,
            log_transition_probability_between_words= -float("inf")#np.log(0.05),
            )
        return score, path

    @classmethod
    def _viterbi_static(
        cls,
        observation_sequence: NDArray[np.float32], 
        log_transition_probabilities: SparseMatrix, 
        multivariate_normals: List[MultivariateNormal],
        initial_likelihood: NDArray[np.float32],
        model_boundaries: ModelBoundary,
        log_transition_probability_between_words: float,
        ) -> Tuple[float, NDArray[np.int8]]:
        
        num_of_states: int = len(multivariate_normals)
        sequence_length: int = observation_sequence.shape[0]
        likelihoods_left = initial_likelihood

        # Init array
        likelihoods_right: NDArray[np.float32] = np.full((num_of_states), -float("inf"), dtype=np.float32)
        tracer: NDArray[np.int8] = np.zeros(
            (sequence_length, num_of_states), 
            dtype=np.int8
            ) - 1 
            # RuntimeWarning: invalid value encountered in cast if full -math.inf

        for time in range(1, sequence_length):

            # First situation: new_state not in word boundaries
            logger.debug(f"Not in model boundaries")
            for new_state in range(num_of_states): # Multi processing able
                if new_state in model_boundaries.lower_boundaries:
                    # Skip when transition from another word is possible
                    continue

                # Find lower boundary for current state
                lower_boundary = model_boundaries.find_lower_boundary(new_state)

                # Find best likelihood
                current_max_likelihood: NDArray[np.float32] = np.full((num_of_states, ), -float("inf"))
                for old_state in range(max(new_state-2, lower_boundary), new_state+1):
                    current_max_likelihood[old_state] = \
                            + log_transition_probabilities[(old_state, new_state)]\
                                + likelihoods_left[old_state]
                max_value: np.float32 = np.max(current_max_likelihood)
                max_index: int = int(np.argmax(current_max_likelihood))

                # Use multivariate normal here so we save compute
                likelihoods_right[new_state] = max_value + multivariate_normals[new_state].log_pdf(observation_sequence[time])

                # Note the best path for each
                tracer[time, new_state] = max_index

            # Second situation: new state in word boundaries
            logger.debug(f"In model boundaries")
            for new_state in model_boundaries.lower_boundaries:
                current_max_likelihood: NDArray[np.float32] = np.full((model_boundaries.num_of_words + 1, ), -float("inf"))
                # Transition from current word
                current_max_likelihood[-1] = \
                            + log_transition_probabilities[(new_state, new_state)]\
                                + likelihoods_left[new_state]

                # Transition from other words
                for index, old_state in enumerate(model_boundaries.upper_boundaries):
                    current_max_likelihood[index] = \
                            + log_transition_probability_between_words\
                                + likelihoods_left[old_state]

                max_value: np.float32 = np.max(current_max_likelihood)
                max_index_upper_boundaries: int = int(np.argmax(current_max_likelihood))
                if max_index_upper_boundaries == model_boundaries.num_of_words:
                    # Simple situation
                    max_index: int = new_state
                else:
                    # Translate back
                    max_index: int = model_boundaries.upper_boundaries[max_index_upper_boundaries]

                # Use multivariate normal here so we save compute
                likelihoods_right[new_state] = max_value + multivariate_normals[new_state].log_pdf(observation_sequence[time])

                # Note the best path for each
                tracer[time, new_state] = max_index

            # Move array for next iteration
            likelihoods_left = likelihoods_right
            likelihoods_right = np.full((num_of_states), -float("inf"), dtype=np.float32)

        # score: float = likelihoods[-1, -1]
        scores: NDArray = likelihoods_left[model_boundaries.upper_boundaries]
        logger.debug(f"Scores are {scores}")
        best_score: float = np.max(scores)
        best_score_index: int = int(np.argmax(scores))
        ## Translate back
        best_score_index = model_boundaries.upper_boundaries[best_score_index]

        # Find the path
        prev_state: int = tracer[-1, best_score_index]
        path: NDArray = np.zeros((sequence_length, ), dtype=np.int8)
        path[-1] = prev_state
        
        for time in range(sequence_length - 2, 0, -1):
            path[time] = prev_state
            prev_state = tracer[time, prev_state]
        return best_score, path
