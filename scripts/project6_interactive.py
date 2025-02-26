import functools
from typing import Dict, List, Tuple
import logging
import concurrent.futures

from numpy.typing import NDArray
import numpy as np
import scipy as sp

from loe_speech_recognition import MFCC, TI_DIGITS_LABELS, HiddenMarkovModelInference, CSVWriter, Segmentation

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def main():
    model_name: str = "big_model_speech_only_continuous"

    # Good Mic Setup
    Segmentation.speech_high_threshold = 384
    Segmentation.speech_low_threshold = 384
    Segmentation.silence_duration_threshold = 0.4
    seg = Segmentation.from_basic(
        sample_rate=16000
    )
    # Words
    seg.main()

    sr, raw_signal = sp.io.wavfile.read("./segment_results/result.wav")
    logger.debug(f"Raw signal is in {raw_signal.dtype} type")
    raw_signal_float: NDArray[np.float32] = raw_signal.astype(np.float32)
    signal = MFCC(raw_signal_float, sr).feature_vector.T

    models_to_load: List[str] = list(TI_DIGITS_LABELS.keys())
    models_to_load.append("S") # Load silence
    hmm_inference = HiddenMarkovModelInference.from_folder(f".cache/{model_name}/", models_to_load)
    hmm_inference._log_transition_probability_between_words = -100

    print(f"Predict result: {hmm_inference.predict(signal)}")

if __name__ == "__main__":
    main()