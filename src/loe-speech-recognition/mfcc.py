import librosa
import numpy as np

def compute_mfcc(signal, sr, n_mfcc=13):
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs) from a signal.

    Args:
        signal (np.ndarray): The input audio signal as a 1D numpy array.
        sr (int): The sampling rate of the audio signal.
        n_mfcc (int): The number of MFCCs to compute (default: 13).

    Returns:
        np.ndarray: A 2D numpy array containing the MFCCs.  Each row represents
                    a time frame, and each column represents an MFCC coefficient.
        np.ndarray: The delta MFCCs (optional, can be returned if needed).
        np.ndarray: The delta-delta MFCCs (optional, can be returned if needed).

    Raises:
        TypeError: If the input signal is not a numpy array.
        ValueError: If the input signal is not 1-dimensional.

    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a numpy array.")
    if signal.ndim != 1:
        raise ValueError("Input signal must be 1-dimensional.")

    # 1. Pre-processing (Optional but recommended)
    # You might want to apply pre-emphasis, windowing, etc. here. Librosa often 
    # handles windowing and STFT implicitly in the mfcc function, but you can 
    # control these steps more directly if needed.

    # 2. Compute the Mel spectrogram
    # Librosa's mfcc function does this internally, but if you need the spectrogram
    # for other purposes, you can calculate it separately:
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40, n_fft=320, hop_length=160, fmin=133.33, fmax=6855.4976)  # Adjust n_mels as needed
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale

    # 3. Compute MFCCs
    mfccs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc)  # Uses log-mel spectrogram internally

    # 4. Compute delta and delta-delta MFCCs (Optional but often useful)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    # print(mfccs.shape)
    # print(delta_mfccs.shape)
    # print(delta2_mfccs.shape)

    return np.concatenate((normalize_mfccs(mfccs), delta_mfccs, delta2_mfccs), axis=0)  # Return MFCCs and their derivatives

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

def main() -> None:
    # Example usage:
    # Load an audio file (replace with your file path)
    file_path = librosa.ex('trumpet')  # Example: Use librosa's example audio
    y, sr = librosa.load(file_path, sr=None)  # y: audio time series, sr: sampling rate

    # Compute MFCCs
    mfccs, delta_mfccs, delta2_mfccs = compute_mfcc(y, sr, n_mfcc=13) #compute 20 MFCCs

    print("MFCCs shape:", mfccs.shape)
    print("Delta MFCCs shape:", delta_mfccs.shape)
    print("Delta-delta MFCCs shape:", delta2_mfccs.shape)

    # You can now use the mfccs (and their deltas) for further analysis or machine learning tasks.

    # Example: Plot the MFCCs (optional)
    import matplotlib.pyplot as plt
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f')
    plt.title('MFCCs')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()