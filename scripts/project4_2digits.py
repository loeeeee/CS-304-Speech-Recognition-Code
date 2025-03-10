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
ti_digits = TIDigits("./ConvertedTIDigits", isLazyLoading=True)
logger.info("Finish loading dataset")

train_dataset = ti_digits.train_dataset

mc = ModelCollection.load_from_files(".cache/big_model")

print(TI_DIGITS_LABELS.keys())
test_from_train_data = [f"{i}{j}" for i, j in itertools.product("".join(list(TI_DIGITS_LABELS.keys())), repeat=2)]
logger.info(f"{test_from_train_data}")
resampled = random.sample(test_from_train_data, 10)

signals = [MFCC(train_dataset.get_combined(test_data, key=2), sample_rate=16000).feature_vector.T for test_data in resampled]
for signal, ground_truth in zip(signals, resampled):
    pred_labels = mc.predict(signal)
    logger.info(f"Predict labels: {"".join(pred_labels)}, ground truth: {ground_truth}")
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for ground_truth, pred_labels in zip(test_from_train_data, executor.map(mc.predict_continuous_controller, signals)):
#         logger.info(f"Predict labels: {"".join(pred_labels)}, ground truth: {ground_truth}")
#         print(pred_labels)
