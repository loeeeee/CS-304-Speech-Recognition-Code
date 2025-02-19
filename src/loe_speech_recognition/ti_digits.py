from dataclasses import dataclass, field
import os
from typing import Dict, Generator, List, Self, Tuple, Any
import logging

import numpy as np
import librosa

logger = logging.getLogger(__name__)

TI_DIGITS_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "O", "Z"]

@dataclass
class DataLoader:
    # Mains
    data: Dict[str, List[np.ndarray]]

    def __post_init__(self) -> None:
        logger.info(f"Create TIDigits Data Loader with {len(self)} labels")
        logger.info(f"Labels: {self.data.keys()}")

    def __iter__(self) -> Generator[Tuple[np.ndarray, str], Any, Any]:
        for k, v in self.data.items():
            for audio_clip in v:
                yield (audio_clip, k)

    def __add__(self, other: Self) -> Self:
        combined_data = self.data
        for k, v in other.data.items():
            if k in combined_data:
                combined_data[k].extend(v)
            else:
                combined_data[k] = v
        return type(self)(combined_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> List[np.ndarray]:
        """
        Return all the data with label, key

        Args:
            key (str): label

        Returns:
            List[np.ndarray]: A list of data
        """
        logger.info(f"Selecting all data with label {key}")
        logger.info(f"Returning {len(self.data[key])} data points")
        return self.data[key]

    @classmethod
    def from_folder_path(cls, folder_path: str, isSingleDigits: bool) -> Self:
        data = {}

        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):  # Walk through all subdirectories
                for filename in filenames:
                    if filename.endswith(".wav") or filename.endswith(".WAV"): # Case-insensitive check
                        filepath = os.path.join(dirpath, filename)
                        try:
                            label = cls.filename_parser(filename)
                            if isSingleDigits and len(label) != 1:
                                logger.debug("Skip non-single-digit data")
                                continue
                            else:
                                y, _ = librosa.load(filepath) # Sample rate is assumed to be 16000
                                if label in data:
                                    data[label].append(y)
                                else:
                                    data[label] = [y]
                        except Exception as e:
                            print(f"Error loading {filepath}: {e}") # Print error for individual files
                            # Optionally, you might want to continue or break here.
                            # For now, we continue to the next file.
        except FileNotFoundError:
            logger.error(f"Error: Root folder '{folder_path}' not found.")
            raise
        except Exception as e:  # Catch other potential errors during directory traversal
            logger.error(f"An error occurred: {e}")
            raise
        
        return cls(data)
    
    @staticmethod
    def filename_parser(file_name: str) -> str:
        result = file_name.split(".")[0][:-1] # 1a 2b
        logger.debug(f"{file_name} is parsed to {result}")
        return result


@dataclass
class TIDigits:
    # Mains
    folder_path: str

    # Settings
    include_adult: bool = field(default=True)
    include_children: bool = field(default=True)
    include_percentage: float = field(default=1.0)
    isSingleDigits: bool = field(default=False)

    # Internals
    _train_dataset: DataLoader = field(init=False)
    _test_dataset: DataLoader = field(init=False)

    def __post_init__(self) -> None:
        # Initialize Dataset
        self._train_dataset = DataLoader({})
        self._test_dataset = DataLoader({})

        # Initialize Adult Dataset
        if self.include_adult:
            adult_folder_path = os.path.join(self.folder_path, "Adults", "TIDIGITS")
            adult_train = DataLoader.from_folder_path(
                os.path.join(adult_folder_path, "TRAIN"),
                self.isSingleDigits
            )
            self._train_dataset += adult_train
            adult_test = DataLoader.from_folder_path(
                os.path.join(adult_folder_path, "TEST"),
                self.isSingleDigits
            )
            self._test_dataset += adult_test
        
        # Initialize Children Dataset
        if self.include_children:
            children_folder_path = os.path.join(self.folder_path, "Children", "TIDIGITS")
            children_train = DataLoader.from_folder_path(
                os.path.join(children_folder_path, "TRAIN"),
                self.isSingleDigits
            )
            self._train_dataset += children_train
            children_test = DataLoader.from_folder_path(
                os.path.join(children_folder_path, "TEST"),
                self.isSingleDigits
            )
            self._test_dataset += children_test

        # Errors
        if not self.include_adult and not self.include_children:
            logger.error(f"Both Adults and Children are not included")
            raise Exception
        
        logger.info("Successfully create TIDigits dataset")

    @property
    def train_dataset(self) -> DataLoader:
        return self._train_dataset

    @property
    def test_dataset(self) -> DataLoader:
        return self._test_dataset

