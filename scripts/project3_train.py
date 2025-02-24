from loe_speech_recognition import TIDigits, HiddenMarkovModelTrainable, MFCC, TI_DIGITS_LABELS

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset

for label in TI_DIGITS_LABELS:
    logger.info("Start computing MFCCs")
    train_dataset_mfccs = MFCC.batch(train_dataset[label], sample_rate=16000)
    logger.info("Finish computing MFCCs")

    # logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Start initialize HMM model from data")
    hmm = HiddenMarkovModelTrainable.from_data(
        label, 
        train_dataset_mfccs,
        num_of_states=5,
        max_iterations=100, 
        isMultiProcessingTraining=True,
        )
    logger.info("Finish initialize HMM model from data")

    logger.info("Start saving HMM model")
    hmm.save(".cache/big_model/")
    logger.info("Finish saving HMM model")
