import queue
import sys
from typing import List, Literal, Self, no_type_check
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

@dataclass
class Segmentation:
    stream: sd.InputStream
    audio_cache: queue.Queue

    # Settings
    frame_size: int = field(default=320)
    speech_high_threshold: int = field(default=512) # Start volume
    speech_low_threshold: int = field(default=64) # Cut volume

    # Internals
    _noise_floor: NoiseFloor = field(default_factory=NoiseFloor) # May use default factory for an init measurement
    _speech_started: bool = field(default=False)
    _speech_ended: bool = field(default=True)
    _results: List[np.ndarray] = field(default_factory=list)
    _per_frame_time: float = field(default=0.02) # Initialize this when main is called

    @no_type_check
    def write_to_wave(self, samples: np.ndarray, name: str) -> None:
        path = os.path.join("segment_results", f"{name}.wav")
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
        self._per_frame_time = 1 / self.stream.samplerate * self.frame_size

        try:
            with self.stream:
                logger.debug("Entering recording stream")
                input("Press any key to start recording")
                self.initialize_noise_floor()
                while True:
                    logger.debug("Start recording")

                    # Routine
                    self.routine()
                    time.sleep(0.2) # Don't draw too much CPU

        except KeyboardInterrupt:
            print("\nGracefully exiting")
            for index, segment in enumerate(self._results):
                self.write_to_wave(segment, str(index).zfill(2))
        ...

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
            if self._speech_started:
                # Detect speech continues
                if self.detect_speech(frame, threshold="low"):
                    self._results[-1] = np.concatenate((self._results[-1], frame))
                    logger.debug("Speech continued")
                else:
                    self._speech_ended = True
                    self._speech_started = False
                    logger.info("Speech stopped")
            else:
                # Detect speech start
                if self.detect_speech(frame, threshold="high"):
                    self._results.append(frame) # Add new words to the results
                    self._speech_started = True
                    self._speech_ended = False
                    logger.info("Speech recognized")
                else:
                    # Update noise floor
                    self._noise_floor.update_noise_floor(frame)
        return

    def detect_speech(self, frames: np.ndarray, threshold: Literal["high", "low"]) -> bool:
        # Detect Speech
        total_energy = np.abs(frames) - self._noise_floor.noise_floor # Subtract noise floor for more consistent performance
        average_energy = np.average(total_energy)

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
    def from_basic(cls, sample_rate: int = 44100, channels: List[int] = [1]) -> Self:
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
        result: Self = cls(stream, audio_cache)

        logger.info(f"Create Segmentation object from basic settings")
        return result


def main() -> None:
    seg = Segmentation.from_basic(
        sample_rate=16000
    )
    seg.speech_high_threshold = 128
    seg.speech_low_threshold = 64
    seg.main()


if __name__ == "__main__":
    main()