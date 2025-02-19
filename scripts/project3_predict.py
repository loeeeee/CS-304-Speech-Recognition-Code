from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC, TI_DIGITS_LABELS

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isSingleDigits=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset

logger.info("Start computing MFCCs")
train_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset["1"][:10]]
test_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset["1"][10:20]]
logger.info("Finish computing MFCCs")

# logging.getLogger().setLevel(logging.DEBUG)
logger.info("Start loading HMM model")
hmm = HiddenMarkovModel.from_file(".cache/0#5#39")
logger.info("Finish loading HMM")

logger.info("Start testing the HMM")
for signal in train_dataset_mfccs:
    score = hmm.predict(signal)
    logger.info(f"Get score {score}")
for signal in test_dataset_mfccs:
    score = hmm.predict(signal)
    logger.info(f"Get score {score}")
logger.info(f"Finish testing the HMM")