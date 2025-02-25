import functools
from typing import Dict, List, Tuple
import logging
import concurrent.futures
import itertools

from matplotlib.pyplot import plot
from numpy.typing import NDArray
import numpy as np

from loe_speech_recognition import TIDigits, MFCC, TI_DIGITS_LABELS, HiddenMarkovModelInference, CSVWriter

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def _make_prediction(hmm_inference: HiddenMarkovModelInference, label_and_signals: Tuple[str, List[NDArray]]) -> Tuple[List[str], List[str]]:
    label, signals = label_and_signals
    logger.info(f"Working on label {label}")
    truth: List[str] = []
    pred: List[str] = []
    for signal in signals:
        pred_label = hmm_inference.predict(signal)
        pred.append(pred_label)
        truth.append(label)
    return truth, pred

def make_prediction(hmm_inference: HiddenMarkovModelInference, labeled_signals: Dict[str, List[NDArray[np.float32]]]) -> Tuple[List[str], List[str]]:
    overall_bar = tqdm(desc="Overall Progress", total=len(labeled_signals), position=0)
    ground_truth: List[str] = []
    prediction: List[str] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for truth, pred in executor.map(
            functools.partial(
                _make_prediction, hmm_inference), 
                labeled_signals.items()):

            ground_truth.extend(truth)
            prediction.extend(pred)
            overall_bar.update()
    return ground_truth, prediction

def accuracy_calculation(ground_truth: List[str], prediction: List[str]) -> None:
    full_match = sum([truth==pred for truth, pred in zip(ground_truth, prediction)]) / len(ground_truth)
    summary = f"In total, {len(ground_truth)} data points tested, the accuracy rate is {full_match * 100: 2f}%"
    logger.info(summary)
    print(summary)
    return

def main():
    logger.info("Start loading dataset")
    ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
    logger.info("Finish loading dataset")

    hmm_inference = HiddenMarkovModelInference.from_folder(".cache/big_model/", list(TI_DIGITS_LABELS.keys()))

    # Train
    train_dataset = ti_digits.train_dataset
    two_digit_signals = train_dataset.get_all_n_digits(2)
    logger.info(f"In total, there are {len(two_digit_signals)} in training dataset")

    two_digit_signals_mfccs = {label: MFCC.batch(signals, sample_rate=16000) for label, signals in two_digit_signals.items()}
    logger.info(f"Finish calculating mfccs")

    logger.info(f"Start making prediction")
    # logging.getLogger().setLevel(logging.DEBUG) # Enable DEBUG
    ground_truth, prediction = make_prediction(hmm_inference, two_digit_signals_mfccs)
    logger.info(f"Finish making prediction")
    accuracy_calculation(ground_truth, prediction)
    # Write to CSV
    csv_writer = CSVWriter(["ground_truth", "prediction"])
    for truth, pred in zip(ground_truth, prediction):
        csv_writer.add_line([truth, pred])
    csv_writer.write(f"./plots/truth_vs_pred.csv")
    # plot_confusion_matrix_from_lists(prediction, ground_truth, class_names=list(set(ground_truth)|set(prediction)), title="ConfusionMatrix2Digits", figsize=(16,12))

    # Test
    test_dataset = ti_digits.test_dataset

if __name__ == "__main__":
    main()