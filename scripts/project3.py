from loe_speech_recognition import TIDigits

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

ti_digits = TIDigits("./ConvertedTIDigits", isSingleDigits=True)

train_dataset = ti_digits.train_dataset

for _, label in train_dataset:
    print(label)