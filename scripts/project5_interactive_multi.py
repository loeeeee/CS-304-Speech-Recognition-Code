import logging

from numpy.typing import NDArray

from loe_speech_recognition import Segmentation, ModelCollection, MFCC

import scipy as sp
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Good Mic Setup
Segmentation.speech_high_threshold = 384
Segmentation.speech_low_threshold = 384
Segmentation.silence_duration_threshold = 0.4
seg = Segmentation.from_basic(
    sample_rate=16000
)
# Words
seg.main()

logging.getLogger().setLevel(logging.DEBUG)

sr, raw_signal = sp.io.wavfile.read("./segment_results/result.wav")
logger.debug(f"Raw signal is in {raw_signal.dtype} type")
raw_signal_float: NDArray[np.float32] = raw_signal.astype(np.float32)
signal = MFCC(raw_signal_float, sr).feature_vector.T

mc = ModelCollection.load_from_files(".cache/big_model", 5, 39)
logger.info("Model loaded")
pred_result = mc.predict_continuous_controller(signal)
print(pred_result)
