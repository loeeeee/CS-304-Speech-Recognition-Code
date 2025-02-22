from dataclasses import dataclass, field
import logging
from typing import List

import librosa
import numpy as np
from numpy._typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class MFCC:
    # Input
    signal: np.ndarray
    sample_rate: int|float

    # Settings
    n_mfcc: int = field(default=13)

    # Internals
    _feature_vector: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.signal, np.ndarray):
            raise TypeError("Input signal must be a numpy array.")
        if self.signal.ndim != 1:
            raise ValueError("Input signal must be 1-dimensional.")

        # Compute MFCC
        mel_spectrogram = librosa.feature.melspectrogram(
            y=self.signal, sr=self.sample_rate, 
            n_mels=40, n_fft=320, hop_length=160, 
            fmin=133.33, fmax=6855.4976)  # Adjust n_mels as needed
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale
        mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=self.sample_rate, n_mfcc=self.n_mfcc)  # Uses log-mel spectrogram internally

        # Compute MFCC Delta
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Compose Feature Vector
        self._feature_vector = np.concatenate((self.normalize_mfccs(mfccs), delta_mfccs, delta2_mfccs), axis=0)
        return

    @property
    def feature_vector(self) -> np.ndarray:
        return self._feature_vector

    @staticmethod
    def normalize_mfccs(mfccs):
        """
        Performs mean subtraction and variance normalization on MFCCs.

        Args:
            mfccs (np.ndarray): The MFCC array (2D).

        Returns:
            np.ndarray: The normalized MFCC array.
        """
        # Calculate the mean across the time axis (axis=0) for each MFCC coefficient
        mfccs_mean = np.mean(mfccs, axis=0, keepdims=True)  # keepdims to maintain shape
        mfccs_std = np.std(mfccs, axis=0, keepdims=True)

        # Subtract the mean and divide by the standard deviation
        normalized_mfccs = (mfccs - mfccs_mean) / (mfccs_std + 1e-8)  # Add a small epsilon to avoid division by zero

        # print(normalized_mfccs.shape)
        return normalized_mfccs

    @classmethod
    def batch(cls, signals: List[NDArray], sample_rate: int) -> List[NDArray[np.float32]]:
        """
        Generate **Transposed** mfcc feature vectors given input signals, 
        row is time, column is features (-1, 39) for example

        Args:
            signals (List[NDArray]): A list of raw signals

        Returns:
            List[NDArray[np.float32]]: A list of transposed time series MFCC feature vectors
        """
        # Really fast, multiprocessing not needed
        return [cls(signal, sample_rate=sample_rate).feature_vector.T for signal in signals]

