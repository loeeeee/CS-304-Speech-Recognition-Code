import numpy as np
import scipy as sp
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(filename='runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Visualization:
    # Settings
    audio_path: str
    frame_size: int = field(default=320)
    hop_length: int = field(default=160)
    n_mels: int = field(default=40)
    n_mfcc: int = field(default=13)
    fmin: float = field(default=133.33)
    fmax: float = field(default=6855.4976)
    fft_window_size: int = field(init=False)
    fft_window: np.ndarray = field(init=False)

    # Internals
    _audio: np.ndarray = field(init=False)
    _sr: float = field(init=False)

    def __post_init__(self) -> None:
        self._audio, self._sr = sf.read(self.audio_path)
        self.fft_window_size = self.frame_size
        self.fft_window = sp.signal.get_window("hamming", self.fft_window_size)
        logger.info(f"Sample Rate: {self._sr}")

    def _fft_2(self):
        n_fft = self.frame_size
        n_overlap = n_fft - self.hop_length
        number_of_frames = (len(self._audio) - n_fft) // self.hop_length + 1
        spectrogram_matrix = []

        for i in range(number_of_frames):
            start = i * self.hop_length
            end = start + n_fft
            frame = self._audio[start:end] * self.fft_window
            spectrum = np.fft.fft(frame, n=n_fft)
            spectrogram_matrix.append(np.abs(spectrum)**2) #Power spectrum

        spectrogram_matrix = np.array(spectrogram_matrix).T
        frequencies = np.fft.fftfreq(n_fft, 1/self._sr)
        times = np.arange(0, len(self._audio)/self._sr, self.hop_length/self._sr)[:spectrogram_matrix.shape[1]]

        return times, frequencies, spectrogram_matrix


    def _fft(self) -> Tuple[np.ndarray, np.ndarray]:
        num_of_windows: int = self._audio.shape[0] // self.frame_size
        # audio_frames = np.copy(self._audio)[:num_of_frames*self.frame_size].reshape((-1, self.frame_size))
        results: List[np.ndarray] = []
        segment_timestamp: List[float] = []
        for index, frames in enumerate(zip(audio_frames[:-3], audio_frames[1:-2], audio_frames[2:])):
            frame = np.concatenate(frames)
            windowed_frame = frame * self.fft_window
            frequency_domain_windowed_frame = np.log(np.abs(sp.fft.fft(windowed_frame)) + 1e-10)
            # print(frequency_domain_windowed_frame.shape) (960,)
            results.append(frequency_domain_windowed_frame)
            segment_timestamp.append(index * (1/self._sr) * self.frame_size)
            # logger.debug(f"Find frequency: {frequency_domain_windowed_frame}")
        result = np.stack(results, axis=1)
        frequency = np.fft.fftfreq(self.frame_size * 3, 1/16000)
        # print(frequency)
        np_segment_timestamp = np.array(segment_timestamp)
        # print(np_segment_timestamp)
        return result, frequency, np_segment_timestamp

    def _cepstrum(self, x, NFFT=None):
        """
        Computes the complex cepstrum of a signal.

        Args:
            x: The input signal.
            NFFT: The FFT size. If None, it defaults to the next power of 2 of len(x).

        Returns:
            A tuple containing:
                - c: The complex cepstrum.
                - q: The quefrency axis.
        """
        if NFFT is None:
            NFFT = 2**int(np.ceil(np.log2(len(x))))

        X = np.fft.fft(x, n=NFFT)
        logX = np.log(np.abs(X))  # Magnitude spectrum in log scale
        c = np.fft.ifft(logX)  # Inverse FFT to get cepstrum
        q = np.fft.fftfreq(NFFT) #Quefrency axis
        return c, q

    def plot_spectrogram(self):
        plt.figure(figsize=(10, 4))
        plt.specgram(self._audio, NFFT=self.frame_size, Fs=self._sr, window=self.fft_window, noverlap=self.frame_size - self.hop_length, scale="dB")
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.show()

    def plot_spectrogram_2(self):
        result, freq, timestamp = self._fft()
        # print(10*np.log10(result))
        print(freq.shape)
        print(timestamp.shape)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(timestamp, freq, result, shading='gouraud') # Convert to dB
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.ylim(20, self._sr/2) # Limit frequency range. Important for log scale
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Log Spectrogram')
        plt.colorbar(label='Intensity (dB)')
        plt.show()

    def plot_spectrogram_3(self):

        times, frequencies, spectrogram_matrix = self._fft_2()

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies[:self.frame_size//2+1], 10 * np.log10(spectrogram_matrix[:self.frame_size//2+1,:]), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram')
        plt.colorbar(label='Power/Frequency [dB]')
        plt.tight_layout()
        plt.show()

    def plot_cepstrum(self):
        Pxx, freqs, bins = mlab.specgram(self._audio, Fs=self._sr, NFFT=self.frame_size*3, noverlap=self.frame_size, window=self.fft_window)

        print(Pxx.shape)
        print(freqs.shape)
        print(bins.shape)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(bins, freqs, 10*np.log10(Pxx), shading='gouraud') # Convert to dB
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.ylim(20, self._sr/2) # Limit frequency range. Important for log scale
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Log Spectrogram')
        plt.colorbar(label='Intensity (dB)')
        plt.show()



    def plot_log_spectrogram(self):
        Pxx, freqs, bins = mlab.specgram(self._audio, Fs=self._sr, NFFT=self.frame_size, noverlap=self.frame_size - self.hop_length, window=self.fft_window)
        print(Pxx.shape)
        print(freqs.shape)
        print(bins.shape)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(bins, freqs, 10*np.log10(Pxx), shading='gouraud') # Convert to dB
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.ylim(20, self._sr/2) # Limit frequency range. Important for log scale
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Log Spectrogram')
        plt.colorbar(label='Intensity (dB)')
        plt.show()

    def plot_cepstrogram_2(self):
        window_size = 3 * self.frame_size
        num_frames = (len(self._audio) - window_size) // (self.frame_size * 2) + 1
        cepstrogram_matrix = []

        # window = np.hamming(window_size)  # Use a Hamming window

        for i in range(num_frames):
            start = i * self.frame_size * 2
            end = start + window_size
            frame = self._audio[start:end] * self.fft_window

            # Compute the complex cepstrum
            spectrum = sp.fft.fft(frame)
            log_spectrum = np.log(np.abs(spectrum) + 1e-10) #avoid log of zero
            cepstrum = np.real(sp.fft.ifft(log_spectrum))

            cepstrogram_matrix.append(cepstrum)

        cepstrogram_matrix = np.array(cepstrogram_matrix).T

        # Plot the cepstrogram
        plt.figure(figsize=(10, 6))
        times = np.arange(0, len(self._audio)/self._sr, self.frame_size * 2/self._sr)[:cepstrogram_matrix.shape[1]]
        quefrencies = np.fft.fftfreq(window_size, 1/self._sr)
        plt.pcolormesh(times, quefrencies[:window_size//2+1], cepstrogram_matrix[:window_size//2+1,:], shading='gouraud', cmap='viridis') #only positive quefrencies
        plt.xlabel("Time (s)")
        plt.ylabel("Quefrency (s)")
        plt.title("Cepstrogram")
        plt.colorbar(label="Magnitude")
        plt.tight_layout()
        plt.show()

    # def plot_cepstrum(self):
    #     """
    #     Plots the cepstrum of a signal.

    #     Args:
    #         x: The input signal.
    #         Fs: The sampling frequency.
    #         title: The title of the plot.
    #     """
    #     c, q = self._cepstrum(self._audio, NFFT=self.frame_size*3)
    #     NFFT = len(c)
    #     q = q * self._sr
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(q, np.abs(c)) # Plot the magnitude cepstrum
    #     plt.xlabel("Quefrency (samples or time if multiplied by Fs)")
    #     plt.ylabel("Magnitude")
    #     plt.title("Cepstrum")
    #     plt.grid(True)
    #     plt.xlim(0, self._sr/2)
    #     plt.show()

if __name__ == "__main__":
    vis = Visualization("./segment_results/result.wav")
    vis._fft()