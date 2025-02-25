from typing import List, Tuple
from loe_speech_recognition import TIDigits, MFCC, TI_DIGITS_LABELS, ModelCollection, DataLoader, plot_confusion_matrix_from_lists

import logging
import concurrent.futures

from tqdm import tqdm

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

def main():
    logger.info("Start loading dataset")
    ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
    logger.info("Finish loading dataset")

    train_dataset = ti_digits.train_dataset
    test_dataset = ti_digits.test_dataset

    mc = ModelCollection.load_from_files(".cache/big_model")

    pred, truth = make_prediction(mc, train_dataset)
    plot_confusion_matrix_from_lists(pred, truth, list(TI_DIGITS_LABELS), title="ConfusionMatrixFromSeenData")
    accuracy = sum([prediction==true_value for prediction, true_value in zip(pred, truth)]) / len(pred)
    print(f"Accuracy of the seen data is {accuracy * 100:2f}%")

    pred, truth = make_prediction(mc, test_dataset)
    plot_confusion_matrix_from_lists(pred, truth, list(TI_DIGITS_LABELS), title="ConfusionMatrixFromUnseenData")
    accuracy = sum([prediction==true_value for prediction, true_value in zip(pred, truth)]) / len(pred)
    logger.info(f"Accuracy of the unseen data is {accuracy * 100:2f}%")
    print(f"Accuracy of the unseen data is {accuracy * 100:2f}%")

if __name__ == "__main__":
    main()