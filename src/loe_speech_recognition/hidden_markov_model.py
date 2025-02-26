from dataclasses import dataclass, field
import os
from typing import Dict, List, Self, Tuple
import logging
import concurrent.futures
import pickle

from numpy.typing import NDArray
import numpy as np
import scipy as sp
from tqdm import tqdm

from .signal import Signal, SortedSignals
from .transition_probability import TransitionProbabilities, LogTransitionProbabilities
from .model_boundary import ModelBoundary

logger = logging.getLogger(__name__)


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
    _log_transition_probs: LogTransitionProbabilities = field(default_factory=LogTransitionProbabilities)

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
        log_trans_probs_file_path: str = os.path.join(model_folder, "log_trans_probs.pickle")
        with open(log_trans_probs_file_path, "wb") as f:
            logger.info(f"Saving log_trans_probs")
            pickle.dump(self._log_transition_probs, f, pickle.HIGHEST_PROTOCOL)

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
        log_trans_probs_file_path: str = os.path.join(model_folder_path, "log_trans_probs.pickle")
        with open(log_trans_probs_file_path, "rb") as f:
            model._log_transition_probs = pickle.load(f)

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
        log_transition_probabilities: LogTransitionProbabilities, 
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
        
        for time in range(sequence_length - 2, -1, -1):
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
    _transition_probs: TransitionProbabilities = field(init=False)

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
        self._log_transition_probs = LogTransitionProbabilities.from_transition_probability(
            self._transition_probs
        )
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
        self._update_middleware_parameters(sorted_signals)
        return

    def _update_middleware_parameters(self, sorted_signals: SortedSignals) -> None:
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

    def _train_external(self, signals: List[Signal]) -> None:
        sorted_signals: SortedSignals = SortedSignals(self.num_of_states)
        for signal in signals:
            sorted_signals.append(signal)
        self._update_middleware_parameters(sorted_signals)
        return

    @classmethod
    def _init_parameters(
        cls,
        sample_signal: NDArray[np.float32], 
        num_of_states: int
        ) -> Tuple[NDArray[np.float32], NDArray[np.float32], TransitionProbabilities]:
        dim_of_features = sample_signal.shape[1]
        # Transition Probabilities
        transition_probs = TransitionProbabilities.from_num_of_states(num_of_states)
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
        covariance = cls._init_covariance(dim_of_features, num_of_states)

        return np.array(means, dtype=np.float32), covariance, transition_probs

    @staticmethod
    def _init_covariance(dim_of_features: int, num_of_states: int) -> NDArray[np.float32]:
        return (np.tile(np.eye(dim_of_features), (num_of_states, 1, 1)) * 0.01).astype(np.float32)

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
    _log_transition_probs: LogTransitionProbabilities = field(init=False)
    _model_boundaries: ModelBoundary = field(init=False)
    _log_transition_probability_between_words: float = field(default=np.log(0.005))

    @classmethod
    def from_folder(cls, folder_path: str, models_to_load: List[str]) -> Self:
        # Create new object
        hmm_inference = cls()

        log_transition_probabilities: LogTransitionProbabilities = LogTransitionProbabilities()
        multivariate_normals: List[MultivariateNormal] = []
        model_labels: List[str] = []
        model_boundaries: ModelBoundary = ModelBoundary()
        # Walk the directory
        for model_folder_name in sorted(os.listdir(folder_path)):
            model_folder_path = os.path.join(folder_path, model_folder_name)
            label = HiddenMarkovModel._model_folder_name_parser(model_folder_path)
            if not (label in models_to_load):
                logger.info(f"Skipping {model_folder_name}, because it is not models to load")
                continue

            # Load the model
            hmm = HiddenMarkovModel.from_folder(model_folder_path)
            
            # Put into structure
            log_transition_probabilities.append(hmm._log_transition_probs)
            multivariate_normals.extend(hmm._multivariate_normals)
            model_boundaries.append(hmm.num_of_states)
            model_labels.append(label)

            logger.info(f"Finish loading all files for {str(label)} model")

        hmm_inference._log_transition_probs = log_transition_probabilities
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
            log_transition_probability_between_words=self._log_transition_probability_between_words, # -float("inf")
            )
        return score, path

    @classmethod
    def _viterbi_static(
        cls,
        observation_sequence: NDArray[np.float32], 
        log_transition_probabilities: LogTransitionProbabilities, 
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
        
        for time in range(sequence_length - 2, -1, -1):
            path[time] = prev_state
            prev_state = tracer[time, prev_state]
        return best_score, path


@dataclass
class HiddenMarkovModelMultiWord(HiddenMarkovModel):
    _model_boundaries: ModelBoundary = field(init=False)

    def get_remuexed_signals(self, mfccs_sequences: List[NDArray[np.float32]]) -> Dict[str, List[Signal]]:
        remuxed_signals: Dict[str, List[Signal]] = {label: [] for label in self._model_boundaries._labels}
        for mfccs in mfccs_sequences:
            _, path = self._viterbi(
                mfccs
            )
            _remuxed_signals: Dict[str, List[Signal]] = self._remux_path_and_signal(
                mfccs, path, self._model_boundaries
            )

            for label, signals in _remuxed_signals.items():
                remuxed_signals[label].extend(signals)
        return remuxed_signals

    @staticmethod
    def _remux_path_and_signal(
        signal: NDArray[np.float32], 
        path: NDArray[np.int8], 
        model_boundaries: ModelBoundary
        ) -> Dict[str, List[Signal]]:
        results: Dict[str, List[Signal]] = {label: [] for label in model_boundaries._labels}
        
        # Loop internals
        last_index: int = 0
        last_state: int = path[last_index]
        last_label = model_boundaries.get_label(last_state)
        for index, state in enumerate(path):
            current_label = model_boundaries.get_label(state)
            if current_label != last_label:
                # This will concat two partial signal with same label into one
                results[last_label].append(
                    Signal(
                        num_of_state=model_boundaries.find_upper_boundary(last_state) \
                            - model_boundaries.find_lower_boundary(last_state),
                        # Get signal
                        signal=signal[last_index: index],
                        # Remove offset
                        path=path[last_index: index] \
                            - model_boundaries.find_lower_boundary(last_state)
                    )
                )
                # Update state and index
                last_index = index
                last_state = path[last_index]
                last_label = model_boundaries.get_label(last_state)
            
        return results

    @classmethod
    def from_parameters(cls, labels: str, trainable_models: Dict[str, HiddenMarkovModelTrainable]) -> Self:
        hmm = cls(labels)
        hmm.isTqdm = False

        log_transition_probabilities: LogTransitionProbabilities = LogTransitionProbabilities()
        multivariate_normals: List[MultivariateNormal] = []
        model_labels: List[str] = []
        model_boundaries: ModelBoundary = ModelBoundary()
        for label in labels:
            log_transition_probability = trainable_models[label]._log_transition_probs
            model_multivariate_normals = trainable_models[label]._multivariate_normals

            log_transition_probabilities.append(log_transition_probability)
            multivariate_normals.extend(model_multivariate_normals)
            model_boundaries.append(len(model_multivariate_normals))
            model_labels.append(label)

        hmm._log_transition_probs = log_transition_probabilities
        hmm._multivariate_normals = multivariate_normals
        logger.info(f"Loading {len(hmm._multivariate_normals)} of multivariate normals for inference")
        # Model boundaries
        model_boundaries.add_model_labels(model_labels)
        hmm._model_boundaries = model_boundaries

        logger.debug(f"Finish reorganize the inference model based on {labels}")
        return hmm


@dataclass
class HiddenMarkovModelTrainContinuous:
    # Internals
    ## Preload
    _trainable_models: Dict[str, HiddenMarkovModelTrainable] = field(default_factory=dict)

    @classmethod
    def from_folder(cls, folder_path: str, models_to_load: List[str]) -> Self:
        # Create new object
        hmm_inference = cls()

        # Walk the directory
        for model_folder_name in sorted(os.listdir(folder_path)):
            model_folder_path = os.path.join(folder_path, model_folder_name)
            label = HiddenMarkovModel._model_folder_name_parser(model_folder_path)
            if not (label in models_to_load):
                logger.info(f"Skipping {model_folder_name}, because it is not models to load")
                continue

            hmm_trainable = HiddenMarkovModelTrainable.from_folder(
                model_folder_path=model_folder_path
                )
            hmm_inference._trainable_models[label] = hmm_trainable
            
            logger.info(f"Finish loading all files for {str(label)} model")
        
        return hmm_inference

    def train(self, labeled_mfccs: Dict[str, List[NDArray[np.float32]]]) -> None:

        remuxed_signals: Dict[str, List[Signal]] = {label: [] for label in labeled_mfccs.keys()}
        for labels, mfccs in labeled_mfccs.items():
            # Reorganize model for training
            hmm = HiddenMarkovModelMultiWord.from_parameters(labels, self._trainable_models)
            
            # Find path
            logger.debug(f"Calculating Viterbi Path")
            _remuxed_signals = hmm.get_remuexed_signals(mfccs)
            
            # Save results
            for _labels, signals in _remuxed_signals.items():
                remuxed_signals[_labels].extend(signals)

        # Update parameters
        for label, signals in remuxed_signals.items():
            ## Update means, covariances
            self._trainable_models[label]._train_external(signals)
            logger.debug(f"Training model {label} with {len(signals)} signals")
            ## Update multivariate normals
            self._trainable_models[label]._update_inference_weights()
        
        return
