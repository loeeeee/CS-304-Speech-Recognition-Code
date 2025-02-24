from typing import List, Tuple
from numpy.typing import NDArray
from loe_speech_recognition import TIDigits, MFCC, TI_DIGITS_LABELS, ModelCollection, DataLoader

import logging
import concurrent.futures

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def make_prediction(mc: ModelCollection, dataset: DataLoader) -> Tuple[List[str], List[str]]:
    overall_bar = tqdm(desc="Overall Progress", total=len(TI_DIGITS_LABELS), position=0)
    truth: List[str] = []
    pred: List[str] = []
    for index, label in enumerate(TI_DIGITS_LABELS):
        logger.info(f"Index {index} is label {label}")
        # Seen data
        dataset_mfccs = MFCC.batch(dataset[label], sample_rate=16000)
        local_bar = tqdm(desc="Local Progress", total=len(dataset_mfccs), position=1, leave=False)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for pred_label in executor.map(mc.predict, dataset_mfccs):
                pred.append(pred_label)
                truth.append(label)
                local_bar.update()
        overall_bar.update()
    return truth, pred

def plot_confusion_matrix_from_lists(predictions, ground_truth, class_names, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plots a confusion matrix from lists of predictions and ground truth.

    Args:
        predictions (list): List of predicted class labels.
        ground_truth (list): List of true class labels.
        class_names (list): List of class names (labels) for the axes.
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap, optional): Matplotlib colormap. Defaults to plt.cm.Blues.
        figsize (tuple, optional): Figure size (width, height). Defaults to (8, 6).
    """

    # Create the confusion matrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(ground_truth, predictions):
        true_index = class_names.index(true_label)
        predicted_index = class_names.index(predicted_label)
        confusion_matrix[true_index, predicted_index] += 1

    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"./plots/confusion_matrix_{title}.png")

def main():
    logger.info("Start loading dataset")
    ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
    logger.info("Finish loading dataset")

    train_dataset = ti_digits.train_dataset
    test_dataset = ti_digits.test_dataset

    mc = ModelCollection.load_from_files(".cache/big_model")

    pred, truth = make_prediction(mc, train_dataset)
    plot_confusion_matrix_from_lists(pred, truth, list(TI_DIGITS_LABELS), title="ConfusionMatrixFromSeenData")

    pred, truth = make_prediction(mc, test_dataset)
    plot_confusion_matrix_from_lists(pred, truth, list(TI_DIGITS_LABELS), title="ConfusionMatrixFromUnseenData")

if __name__ == "__main__":
    main()