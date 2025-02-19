from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC

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
logger.info("Finish computing MFCCs")

# logging.getLogger().setLevel(logging.DEBUG)
logger.info("Start initialize HMM model from data")
hmm = HiddenMarkovModel.from_data("0", 5, train_dataset_mfccs)
logger.info("Finish initialize HMM model from data")

logger.info("Start testing HMM model")
logger.info("Finish testing HMM model")
