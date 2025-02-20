import logging

from numpy.typing import NDArray

from loe_speech_recognition import Segmentation, ModelCollection

import librosa

logging.basicConfig(filename='../../runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Good Mic Setup
Segmentation.speech_high_threshold = 2048
Segmentation.speech_low_threshold = 1024
Segmentation.silence_duration_threshold = 0.2
seg = Segmentation.from_basic(
    sample_rate=16000
)
# Words
seg.main()

signal, sr = librosa.load("./segment_results/result.wav", sr=None)

mc = ModelCollection.load_from_files("./cache/big_model", 5, 39)
pred_result = mc.predict_continuous_controller(signal)
print(pred_result)
