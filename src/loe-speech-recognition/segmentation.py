import queue
import sys
from typing import ClassVar, List, Literal, no_type_check
from dataclasses import dataclass, field
import itertools
import os
import wave
import logging
import time

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(filename='runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class NoiseFloor:
    background_samples: List[np.ndarray] = field(default_factory=list)

    # Settings
    num_of_samples: int = field(default=5)

    # Internal
    _noise_floor: int = field(default=0)

    def update_noise_floor(self, samples: np.ndarray) -> int:
        self.background_samples.append(samples)
        if len(self.background_samples) > self.num_of_samples:
            self.background_samples.pop(0)
        
        self._calculate_noise_floor()

        return self._noise_floor

    def _calculate_noise_floor(self) -> None:
        noise_floor = 0
        multiplier: int = 0
        for index, samples in enumerate(reversed(self.background_samples)):
            multiplier += self.num_of_samples - index
            noise_floor += (self.num_of_samples - index) * np.average(np.abs(samples))

        self._noise_floor: int = int(noise_floor / multiplier)
        logger.debug(f"Current noise floor is {self._noise_floor}")
        return

    def __str__(self) -> str:
        return str(self._noise_floor)

    @property
    def noise_floor(self) -> int:
        return self._noise_floor

class _SegmentationDone(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@dataclass
class _SpeechEndCounter:
    frame_count_threshold: int

    # Internals
    _counter: int = field(default=0)

    def __post_init__(self) -> None:
        logger.info(f"Empty frame count threshold is {self.frame_count_threshold}")

    def _check(self) -> None:
        if self._counter >= self.frame_count_threshold:
            logger.info(f"Speech ending empty frame threshold meet")
            raise _SegmentationDone

    def no_speech(self) -> None:
        self._counter += 1
        logger.debug(f"Counter is now {self._counter}")
        self._check()

    def has_speech(self) -> None:
        # Reset counter when speech detected
        self._counter = 0
        logger.debug("Counter reset")

@dataclass
class Segmentation:
    stream: sd.InputStream
    audio_cache: queue.Queue
    save_path: str = field(default="./segment_results")

    # Settings
    frame_size: ClassVar[int] = field(default=320)
    speech_high_threshold: ClassVar[int] = field(default=512) # Start volume
    speech_low_threshold: ClassVar[int] = field(default=64) # Cut volume
    silence_duration_threshold: ClassVar[float] = field(default=0.1)

    # Internals
    _noise_floor: NoiseFloor = field(default_factory=NoiseFloor) # May use default factory for an init measurement
    # This flag means that a speech signal is detected, and goes on continuously. This flag will be reset when signal drop below low threshold
    _isSpeechBetweenHighLowThreshold: bool = field(default=False) 
    # This flag means that a speech signal is detected in this input session. This flag will only be reset on next input session
    _isSpeechEverHighThreshold: bool = field(default=False)
    # This flag means that a speech signal is no longer detected. This flag will be reset when a signal drop below low threshold
    _isSpeechBelowLowThreshold: bool = field(default=True)
    _speech_ended_cnt: _SpeechEndCounter = field(init=False) # Count how many frames with no speech
    _results: List[np.ndarray] = field(default_factory=list)
    _per_frame_time: float = field(init=False) # Initialize this when main is called
    _maximum_silence_frames: int = field(init=False)

    def __post_init__(self) -> None:
        self._per_frame_time = 1 / self.stream.samplerate * self.frame_size
        self._maximum_silence_frames = int(self.silence_duration_threshold / self._per_frame_time)
        self._speech_ended_cnt = _SpeechEndCounter(self._maximum_silence_frames)
        logger.info(f"Single frame is {self._per_frame_time}s")
        logger.info(f"Maximum silence frames count is {self._maximum_silence_frames}")
        return

    @no_type_check
    def write_to_wave(self, samples: np.ndarray, name: str) -> None:
        path = os.path.join(self.save_path, f"{name}.wav")
        sample_rate: int = self.stream.samplerate
        with wave.open(path, "wb") as wav:
            wav.setframerate(sample_rate)
            wav.setnchannels(self.stream.channels)
            wav.setsampwidth(2) # 16 bit
            wav.writeframes(samples.tobytes())

        logger.debug(f"Save {samples.shape[0] * self._per_frame_time / self.frame_size:.2f}s audio to {name}.wav")
        return

    def main(self) -> None:
        """The main loop where hit-to-talk happens"""
        try:
            with self.stream:
                logger.debug("Entering recording stream")
                input("Press any key to start recording")
                self._isSpeechEverHighThreshold = False # Reset flag
                # Clean up cache and set noise floor before starting
                self.initialize_noise_floor()
                print("Recording started")
                while True:
                    # Routine
                    time.sleep(self.silence_duration_threshold + self._per_frame_time) # Don't draw too much CPU
                    self.routine()

        except (KeyboardInterrupt, _SegmentationDone):
            print("\nGracefully exiting")
        
        if self._results:
            result = np.concatenate(self._results[:-self._speech_ended_cnt.frame_count_threshold])
            self.write_to_wave(result, "result")
        else:
            logger.warning("No results from segmentation")
        return

    def routine(self) -> None:
        """Actually do things here"""
        # Get all cached signal
        audio = self.get_all_frames_from_queue(self.audio_cache)

        num_of_frames: int = audio.shape[0] // self.frame_size
        logger.debug(f"Getting {num_of_frames} frames")

        # Construct an iterable for all audio
        trimmed_audio = audio[:self.frame_size*num_of_frames]
        iter_audio_frames = itertools.chain.from_iterable((trimmed_audio.reshape((-1, self.frame_size)), [audio[self.frame_size*num_of_frames:]]))

        for frame in iter_audio_frames:
            if self._isSpeechEverHighThreshold:
                self._results.append(frame)

            if self._isSpeechBetweenHighLowThreshold:
                # Detect speech continues
                if self.detect_speech(frame, threshold="low"):
                    # Speech remain between high low threshold
                    logger.debug("Speech continued")
                    self._speech_ended_cnt.has_speech()
                else:
                    # Speech drop below low threshold
                    logger.info("Speech stopped")
                    self._isSpeechBetweenHighLowThreshold = False
                    self._speech_ended_cnt.no_speech()
                    # logger.debug(f"Current frame: {frame}")
            else:
                # Detect speech start
                if self.detect_speech(frame, threshold="high"):
                    # Speech detected
                    logger.info("Speech recognized")
                    self._isSpeechBetweenHighLowThreshold = True
                    self._isSpeechEverHighThreshold = True
                    self._speech_ended_cnt.has_speech()
                else:
                    if self._isSpeechEverHighThreshold:
                        logger.info("Speech no longer detected")
                        self._speech_ended_cnt.no_speech()
                    # Update noise floor

        # logger.debug(f"Result: {self._results}")
        return

    def detect_speech(self, frames: np.ndarray, threshold: Literal["high", "low"]) -> bool:
        # Detect Speech
        total_energy = np.abs(frames) # - self._noise_floor.noise_floor) # Subtract noise floor for more consistent performance
        average_energy = np.average(total_energy)
        logger.debug(f"Average energy: {average_energy}")

        if threshold == "high" and average_energy > self.speech_high_threshold:
                return True
        elif threshold == "low" and average_energy > self.speech_low_threshold:
                return True

        return False

    def initialize_noise_floor(self) -> None:
        samples = self.get_all_frames_from_queue(self.audio_cache)
        self._noise_floor.update_noise_floor(samples)
        logger.info(f"Noise floor is initialized to {self._noise_floor}")

    # ---------------
    @staticmethod
    def get_all_frames_from_queue(cache: queue.Queue) -> np.ndarray:
        results: np.ndarray = cache.get().reshape(-1)
        try:
            while True:
                results = np.concatenate((results, cache.get_nowait().reshape(-1)))
        except queue.Empty:
            pass
        return results

    # ---------------
    @classmethod
    def from_basic(cls, sample_rate: int = 44100, channels: List[int] = [1], save_path: str = "./segment_results") -> "Segmentation":
        audio_cache: queue.Queue = queue.Queue()
        mapping = [c - 1 for c in channels]  # Channel numbers start with 1

        def audio_callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            # Fancy indexing with mapping creates a (necessary!) copy:
            audio_cache.put(indata[::1, mapping])
        
        stream = sd.InputStream(
            channels=max(channels),
            samplerate=sample_rate, 
            callback=audio_callback,
            dtype=np.int16,
            ) # Specify the 16-bit data type
        result = cls(stream, audio_cache, save_path)

        logger.info(f"Create Segmentation object from basic settings")
        return result


def main() -> None:
    # Good Mic Setup
    Segmentation.speech_high_threshold = 128
    Segmentation.speech_low_threshold = 16
    Segmentation.silence_duration_threshold = 0.2
    seg = Segmentation.from_basic(
        sample_rate=16000
    )
    # Words
    seg.main()


if __name__ == "__main__":
    main()