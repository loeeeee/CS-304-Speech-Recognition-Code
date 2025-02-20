from numpy.typing import NDArray
from loe_speech_recognition import TIDigits, HiddenMarkovModel, MFCC, TI_DIGITS_LABELS, ModelCollection

import logging
import concurrent.futures
import itertools
import random

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

mc = ModelCollection.load_from_files(".cache/big_model", 5, 39)

print(TI_DIGITS_LABELS.keys())
test_from_train_data = [f"{d0}{d1}{d2}{d3}" for d0, d1, d2, d3 in itertools.product("".join(list(TI_DIGITS_LABELS.keys())), repeat=4)]
logger.info(f"{test_from_train_data}")

signals = [MFCC(train_dataset.get_combined(test_data, key=2), sample_rate=16000).feature_vector.T for test_data in random.sample(test_from_train_data, 50)]
with concurrent.futures.ProcessPoolExecutor() as executor:
    for ground_truth, pred_labels in zip(test_from_train_data, executor.map(mc.predict_phone_controller, signals)):
        logger.info(f"Predict labels: {"".join(pred_labels)}, ground truth: {ground_truth}")
        print(pred_labels)
