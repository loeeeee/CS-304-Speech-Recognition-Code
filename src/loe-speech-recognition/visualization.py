from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import get_window
import numpy as np
import librosa
import librosa.display


@dataclass
class Visualization:
    # Settings
    audio_path: str
    n_fft: int = field(default=320)
    hop_length: int = field(default=160)

    # Internals
    _audio: np.ndarray = field(init=False)
    _sr: float = field(init=False)

    def __post_init__(self) -> None:
        self._audio, self._sr = librosa.load(self.audio_path)

    def plot_spectrogram(self, title="Spectrogram"):
        """
        Generates and plots a spectrogram of an audio clip.

        Args:
            audio (np.ndarray): The audio signal as a numpy array.
            sr (int): The sample rate of the audio signal.
            title (str, optional): The title of the plot. Defaults to "Spectrogram".
            n_fft (int, optional): The length of the FFT window. Defaults to 2048.
            hop_length (int, optional): The hop length between FFT windows. Defaults to 512.
        """
        spectrogram = librosa.stft(self._audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_db, sr=self._sr, hop_length=self.hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    