from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC, TI_DIGITS_LABELS, ModelCollection

import logging
import concurrent.futures

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

logger.info("Start loading dataset")
ti_digits = TIDigits("./ConvertedTIDigits", isSingleDigits=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset

mc = ModelCollection.load_from_files(".cache/small_model", 5, 39)

train_data_confusion_matrix = np.zeros((len(TI_DIGITS_LABELS), len(TI_DIGITS_LABELS)))
test_data_confusion_matrix = np.zeros((len(TI_DIGITS_LABELS), len(TI_DIGITS_LABELS)))
overall_bar = tqdm(desc="Overall Progress", total=len(TI_DIGITS_LABELS), position=1)
for index, label in enumerate(TI_DIGITS_LABELS):
    # Seen data
    train_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset[label][:10]]
    local_bar = tqdm(desc="Local Progress", total=len(train_dataset_mfccs), position=0)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for pred_label in executor.map(mc.predict, train_dataset_mfccs):
            pred_index = TI_DIGITS_LABELS[pred_label]
            train_data_confusion_matrix[index, pred_index] += 1
            local_bar.update()

    # Unseen data
    test_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset[label][10:20]]
    local_bar = tqdm(desc="Local Progress", total=len(test_dataset_mfccs), position=0)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for pred_label in executor.map(mc.predict, test_dataset_mfccs):
            pred_index = TI_DIGITS_LABELS[pred_label]
            test_data_confusion_matrix[index, pred_index] += 1
            local_bar.update()
    
    overall_bar.update()

logger.info(f"Confusion matrix when prediction on known data:")
logger.info(f"{train_data_confusion_matrix}")
logger.info(f"Confusion matrix when prediction on unknown data:")
logger.info(f"{test_data_confusion_matrix}")

