from loe_speech_recognition import TIDigits, HiddenMarkovModelTrainable, MFCC, TI_DIGITS_LABELS, SignalSeparation

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset
signal_separation = SignalSeparation(
    sample_rate=16000,
    speech_high_threshold=0.06,
    speech_low_threshold=0.01,
)

for label in TI_DIGITS_LABELS:
    logger.info("Starting removing silence")
    train_dataset_speech_only = signal_separation.remove_empty_batch(train_dataset[label])
    logger.info("Finish removing silence")

    logger.info("Start computing MFCCs")
    train_dataset_mfccs = MFCC.batch(train_dataset_speech_only, sample_rate=16000)
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
    hmm.save(".cache/big_model_speech_only/")
    logger.info("Finish saving HMM model")


logger.info("Start initialize HMM model from silence")
hmm = HiddenMarkovModelTrainable.from_data(
    "S", 
    MFCC.batch(signal_separation.get_all_noises(), sample_rate=16000),
    num_of_states=5,
    max_iterations=100, 
    isMultiProcessingTraining=True,
    )
logger.info("Finish initialize HMM model from silence")

logger.info("Start saving HMM model")
hmm.save(".cache/big_model_speech_only/")
logger.info("Finish saving HMM model")