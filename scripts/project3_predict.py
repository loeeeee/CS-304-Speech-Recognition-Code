from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC, TI_DIGITS_LABELS, ModelCollection

import logging

import numpy as np
import matplotlib.pyplot as plt

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
for index, label in enumerate(TI_DIGITS_LABELS):
    # Seen data
    train_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset[label][:10]]
    for signal in train_dataset_mfccs:
        pred_label = mc.predict(signal)
        pred_index = TI_DIGITS_LABELS[pred_label]
        train_data_confusion_matrix[index: pred_index] += 1

    # Unseen data
    test_dataset_mfccs = [MFCC(i, sample_rate=16000).feature_vector.T for i in train_dataset[label][10:20]]
    for signal in test_dataset_mfccs:
        pred_label = mc.predict(signal)
        test_data_confusion_matrix[label: pred_label] += 1

