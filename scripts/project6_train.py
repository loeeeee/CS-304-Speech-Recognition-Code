from typing import Dict, List
import logging

from numpy.typing import NDArray
import numpy as np

from loe_speech_recognition import TIDigits, MFCC, TI_DIGITS_LABELS, HiddenMarkovModelTrainContinuous

logger = logging.getLogger(__name__)
logging.basicConfig(filename='./runtime.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def main():
    sample_rate: int = 16000
    model_name: str = "big_model_speech_only"

    logger.info("Start loading dataset")
    ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
    logger.info("Finish loading dataset")

    models_to_load: List[str] = list(TI_DIGITS_LABELS.keys())
    models_to_load.append("S") # Load silence
    hmm_train = HiddenMarkovModelTrainContinuous.from_folder(f".cache/{model_name}/", models_to_load)
    hmm_train.isMultiProcessing = True

    train_dataset = ti_digits.train_dataset
    labeled_mfccs: Dict[str, List[NDArray[np.float32]]] = {}
    for i in range(2, 8):
        logger.info(f"Adding dataset of {i} digits")
        dataset = train_dataset.get_all_n_digits(i)
        for label, signals in dataset.items():
            labeled_mfccs[label] = MFCC.batch(signals, sample_rate=sample_rate)
    logger.info(f"Total training set size is {len(labeled_mfccs)}")
    # labeled_mfccs = {label: labeled_mfccs[label] for label in ["4Z2Z1", "943ZZ", "22114"]}
    try:
        hmm_train.train(labeled_mfccs=labeled_mfccs, max_iterations=200)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        hmm_train.save(f".cache/{model_name}_continuous_2")


if __name__ == "__main__":
    main()