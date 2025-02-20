from numpy.typing import NDArray
from loe_speech_recognition import TIDigits, Segmentation, MFCC, TI_DIGITS_LABELS, ModelCollection

import logging
import concurrent.futures
import itertools

import numpy as np
import scipy as sp
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isSingleDigits=True)
logger.info("Finish loading dataset")

# Good Mic Setup
Segmentation.speech_high_threshold = 2048
Segmentation.speech_low_threshold = 1024
Segmentation.silence_duration_threshold = 0.2
seg = Segmentation.from_basic(
    sample_rate=16000
)
# Words
seg.main()

sr, raw_signal = sp.io.wavfile.read("./segment_results/result.wav")
logger.debug(f"Raw signal is in {raw_signal.dtype} type")
raw_signal_float: NDArray[np.float32] = raw_signal.astype(np.float32)
signal = MFCC(raw_signal_float, sr).feature_vector.T

mc = ModelCollection.load_from_files(".cache/big_model", 5, 39)

print(mc.predict_phone_controller(signal))

