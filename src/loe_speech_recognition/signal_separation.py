import queue
import sys
from typing import ClassVar, List, Literal, Tuple, no_type_check
from dataclasses import dataclass, field
import itertools
import os
import wave
import logging
import time

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class _SpeechEndCounter:
    class _SegmentationDone(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)

    frame_count_threshold: int

    # Internals
    _counter: int = field(default=0)

    def __post_init__(self) -> None:
        logger.info(f"Empty frame count threshold is {self.frame_count_threshold}")

    def _check(self) -> None:
        if self._counter >= self.frame_count_threshold:
            logger.info(f"Speech ending empty frame threshold meet")
            raise self._SegmentationDone

    def no_speech(self) -> None:
        self._counter += 1
        logger.debug(f"Counter is now {self._counter}")
        self._check()

    def has_speech(self) -> None:
        # Reset counter when speech detected
        self._counter = 0
        logger.debug("Counter reset")


@dataclass
class SignalSeparation:
    class FailToProcess(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__(*args)
            logger.error("Failed to process signal")

    # Settings
    sample_rate: int = field(default=16000)
    frame_time: float = field(default=0.01)
    speech_high_threshold: float = field(default=0.08)
    speech_low_threshold: float = field(default=0.01)
    silence_duration_threshold: float = field(default=0.02)

    # Internals
    _noises: List[NDArray[np.float32]] = field(default_factory=list)
    _max_volume: float = field(init=False)
    _result: List[NDArray[np.float32]] = field(default_factory=list)
    _noise: List[NDArray[np.float32]] = field(default_factory=list)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_time)

    @property
    def maximum_silence_frames(self) -> int:
        return int(self.silence_duration_threshold / self.frame_time)

    @property
    def _speech_high_threshold(self) -> float:
        return self.speech_high_threshold * self._max_volume

    @property
    def _speech_low_threshold(self) -> float:
        return self.speech_low_threshold * self._max_volume

    def remove_empty_batch(self, signals: List[NDArray[np.float32]]) -> List[NDArray[np.float32]]:
        results = []
        for signal in signals:
            try:
                results.append(self.remove_empty(signal))
            except self.FailToProcess:
                logger.warning(f"Signal with property: length {signal.shape[0]}, max {np.abs(np.max(signal))} failed")
                continue
        return results

    def remove_empty(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        try:
            self._remove_empty(signal)
        except _SpeechEndCounter._SegmentationDone:
            self._noises.append(np.concatenate(self._noise, dtype=np.float32))
            self._noise = []
            result = np.concatenate(self._result, dtype=np.float32)
            if len(self._result) < 9: # This threshold is based on MFCC
                logger.error(f"Resulting audio clip too short, {result}")
                raise self.FailToProcess
            return result
        # logger.error(f"Failed to segment signal, {signal.tolist()}")
        raise self.FailToProcess

    def _remove_empty(self, signal: NDArray[np.float32]) -> None:
        num_of_frames: int = signal.shape[0] // self.frame_size
        logger.debug(f"Getting {num_of_frames} frames")
        self._max_volume = np.max(np.abs(signal)).astype(float)

        # Construct an iterable for all audio
        trimmed_audio = signal[:self.frame_size*num_of_frames]
        iter_audio_frames = itertools.chain.from_iterable((trimmed_audio.reshape((-1, self.frame_size)), [signal[self.frame_size*num_of_frames:]]))
        
        speech_ended_cnt = _SpeechEndCounter(self.maximum_silence_frames)

        # Empty the result
        self._result = []
        isSpeechEverHighThreshold: bool = False
        isSpeechBetweenHighLowThreshold: bool = False
        for frame_index, frame in enumerate(iter_audio_frames):
            if isSpeechBetweenHighLowThreshold:
                # Detect speech continues
                if self.detect_speech(frame, threshold="low"):
                    # Speech remain between high low threshold
                    logger.debug("Speech continued")
                    speech_ended_cnt.has_speech()
                else:
                    # Speech drop below low threshold
                    logger.info(f"Speech stopped at {frame_index * self.frame_time}")
                    isSpeechBetweenHighLowThreshold = False
                    speech_ended_cnt.no_speech()
                    # logger.debug(f"Current frame: {frame}")
            else:
                # Detect speech start
                if self.detect_speech(frame, threshold="high"):
                    # Speech detected
                    logger.info(f"Speech recognized at {frame_index * self.frame_time}")
                    isSpeechBetweenHighLowThreshold = True
                    isSpeechEverHighThreshold = True
                    speech_ended_cnt.has_speech()
                else:
                    self._noise.append(frame)
                    if isSpeechEverHighThreshold:
                        logger.info("Speech no longer detected")
                        speech_ended_cnt.no_speech()

            if isSpeechEverHighThreshold:
                self._result.append(frame)
        
        return

    def get_all_noises(self) -> List[NDArray[np.float32]]:
        logger.info("Get all noises")
        return self._noises

    def detect_speech(self, frames: NDArray, threshold: Literal["high", "low"]) -> bool:
        # Detect Speech
        total_energy = np.abs(frames)
        average_energy = np.average(total_energy)
        logger.debug(f"Average energy: {average_energy}")

        if threshold == "high" and average_energy > self._speech_high_threshold:
                return True
        elif threshold == "low" and average_energy > self._speech_low_threshold:
                return True

        return False
        