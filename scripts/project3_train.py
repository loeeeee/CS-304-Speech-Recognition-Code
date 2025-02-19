from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

# Constant
TI_DIGITS_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "O", "Z"]

logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isSingleDigits=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset

for label in TI_DIGITS_LABELS:
    logger.info("Start computing MFCCs")
    train_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset[label][:10]]
    logger.info("Finish computing MFCCs")

    # logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Start initialize HMM model from data")
    hmm = HiddenMarkovModel.from_data(label, 5, train_dataset_mfccs, k_means_max_iteration=100)
    logger.info("Finish initialize HMM model from data")

    logger.info("Start saving HMM model")
    hmm.save(".cache/small_model/")
    logger.info("Finish saving HMM model")
