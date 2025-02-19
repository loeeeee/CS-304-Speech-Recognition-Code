from numpy.typing import NDArray
from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC, TI_DIGITS_LABELS, ModelCollection

import logging
import concurrent.futures

import numpy as np
from tqdm import tqdm
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

def plot_confusion_matrix(confusion_matrix: NDArray, name: str):
    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(TI_DIGITS_LABELS)), labels=TI_DIGITS_LABELS,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(TI_DIGITS_LABELS)), labels=TI_DIGITS_LABELS)

    # Loop over data dimensions and create text annotations.
    for i in range(len(TI_DIGITS_LABELS)):
        for j in range(len(TI_DIGITS_LABELS)):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.savefig(f"./plots/confusion_matrix_{name}.png")


logger.info(f"Confusion matrix when prediction on known data:")
logger.info(f"{train_data_confusion_matrix}")
plot_confusion_matrix(train_data_confusion_matrix, "known_data_small_model")
logger.info(f"Confusion matrix when prediction on unknown data:")
logger.info(f"{test_data_confusion_matrix}")
plot_confusion_matrix(test_data_confusion_matrix, "unknown_data_small_model")

