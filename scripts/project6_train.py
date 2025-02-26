import functools
from typing import Dict, List, Tuple
import logging
import concurrent.futures

from numpy.typing import NDArray
import numpy as np

from loe_speech_recognition import TIDigits, MFCC, TI_DIGITS_LABELS, HiddenMarkovModelTrainContinuous

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def main():
    N: int = 7
    sample_rate: int = 16000
    model_name: str = "big_model_speech_only"

    logger.info("Start loading dataset")
    ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
    logger.info("Finish loading dataset")

    models_to_load: List[str] = list(TI_DIGITS_LABELS.keys())
    models_to_load.append("S") # Load silence
    hmm_train = HiddenMarkovModelTrainContinuous.from_folder(f".cache/{model_name}/", models_to_load)
    hmm_train.isMultiProcessing = False

    train_dataset = ti_digits.train_dataset
    labeled_mfccs = {label: MFCC.batch(signals, sample_rate=sample_rate) for label, signals in train_dataset.get_all_n_digits(N).items()}
    hmm_train.train(labeled_mfccs=labeled_mfccs)
    hmm_train.save(f".cache/{model_name}_continuous")


if __name__ == "__main__":
    main()