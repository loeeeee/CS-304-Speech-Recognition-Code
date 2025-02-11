import logging

from loe_speech_recognition import Segmentation

logging.basicConfig(filename='../../runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Good Mic Setup
Segmentation.speech_high_threshold = 128
Segmentation.speech_low_threshold = 16
Segmentation.silence_duration_threshold = 0.2
seg = Segmentation.from_basic(
    sample_rate=16000
)
# Words
seg.main()
