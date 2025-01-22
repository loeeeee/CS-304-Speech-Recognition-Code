from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import get_window
import numpy as np
import librosa
import librosa.display

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Visualization:
    # Settings
    audio_path: str
    n_fft: int = field(default=320)
    hop_length: int = field(default=160)
    n_mels: int = field(default=40)
    n_mfcc: int = field(default=13)
    fmin: float = field(default=133.33)
    fmax: float = field(default=6855.4976)

    # Internals
    _audio: np.ndarray = field(init=False)
    _sr: float = field(init=False)

    def __post_init__(self) -> None:
        self._audio, self._sr = librosa.load(self.audio_path, sr=None)
        logger.info(f"Sample Rate: {self._sr}")

    def plot_spectrogram(self, title="Spectrogram"):
        spectrogram = librosa.stft(self._audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=self._sr, hop_length=self.hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_mel_spectrogram(self):
        window = get_window("triang", self.n_fft)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=self._audio, sr=self._sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels, 
            fmin=self.fmin, fmax=self.fmax,
            window=window
            )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=self._sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    def plot_mfcc(self):
        """
        Generates and plots the log Mel spectrogram and MFCCs of an audio clip with custom Mel filter settings.
        """
        # Compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=self._audio, sr=self._sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        
        # Convert to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=self._sr, n_mfcc=self.n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(delta_mfccs)

        # Plot MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=self._sr, hop_length=self.hop_length, x_axis='time')
        plt.colorbar()
        plt.title(f"MFCC")
        plt.tight_layout()
        plt.show()

        # Plot delta MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(delta_mfccs, sr=self._sr, hop_length=self.hop_length, x_axis='time')
        plt.colorbar()
        plt.title(f"Delta MFCC")
        plt.tight_layout()
        plt.show()

        # Plot delta delta MFCCs
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(delta_delta_mfccs, sr=self._sr, hop_length=self.hop_length, x_axis='time')
        plt.colorbar()
        plt.title(f"Delta Delta MFCC")
        plt.tight_layout()
        plt.show()

    def main(self) -> None:
        self.plot_spectrogram()
        self.plot_mel_spectrogram()
        self.plot_mfcc()
