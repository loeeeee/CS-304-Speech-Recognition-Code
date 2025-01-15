import queue
import sys
from typing import List, Literal, Self, no_type_check
from dataclasses import dataclass, field
import itertools
import os
import wave
import logging

import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Segmentation:
    stream: sd.InputStream
    audio_cache: queue.Queue

    # Settings
    frame_size: int = field(default=320)
    speech_high_threshold: int = field(default=512) # Start volume
    speech_low_threshold: int = field(default=64) # Cut volume

    # Internals
    _noise_floor: int = field(default=0) # May use default factory for an init measurement
    _speech_started: bool = field(default=False)
    _speech_ended: bool = field(default=False)
    _results: List[np.ndarray] = field(default_factory=list)
    _per_frame_time: float = field(default=0.02) # Initialize this when main is called

    @no_type_check
    def write_to_wave(self, frames: np.ndarray, name: str) -> None:
        path = os.path.join("segment_results", f"{name}.wav")
        with wave.open(path, "wb") as wav:
            wav.setframerate(self.stream.samplerate)
            wav.setnchannels(self.stream.channels)
            wav.setsampwidth(2) # 16 bit
            wav.writeframes(frames.tobytes())

        logger.debug(f"Save to {name}.wav")
        return

    def main(self) -> None:
        """The main loop where hit-to-talk happens"""
        self._per_frame_time = 1 / self.stream.samplerate * self.frame_size

        try:
            with self.stream:
                while True:
                    input("Press any key to start recording")
                    
                    # Routine

                    pass 
        except KeyboardInterrupt:
            print("Gracefully exiting")
            for index, segment in enumerate(self._results):
                self.write_to_wave(segment, str(index).zfill(2))
        ...

    def routine(self) -> None:
        """Actually do things here"""
        # Get all cached signal
        audio = self.get_all_frames_from_queue(self.audio_cache)

        num_of_frames: int = audio.shape[0] // self.frame_size

        # Construct an iterable for all audio
        trimmed_audio = audio[:self.frame_size*num_of_frames]
        iter_audio_frames = itertools.chain.from_iterable((trimmed_audio.reshape((-1, self.frame_size)), [audio[self.frame_size*num_of_frames:]]))

        for frame in iter_audio_frames:
            if self._speech_started:
                # Detect speech continues
                if self.detect_speech(frame, threshold="low"):
                    self._results[-1] = np.concatenate((self._results[-1], frame))
                    pass
                else:
                    self._speech_ended = True
                    pass
            else:
                # Detect speech start
                if self.detect_speech(frame, threshold="high"):
                    self._results.append(frame) # Add new words to the results
                    self._speech_started = True
        return

    def detect_speech(self, frames: np.ndarray, threshold: Literal["high", "low"]) -> bool:
        # Detect Speech
        abs_total_frame = np.abs(frames) - self._noise_floor
        average_energy = np.average(abs_total_frame)

        if threshold == "high" and average_energy > self.speech_high_threshold:
                return True
        elif threshold == "low" and average_energy > self.speech_low_threshold:
                return True

        return False

    # ---------------
    @staticmethod
    def get_noise_floor(frames: np.ndarray) -> int:
        return int(np.average(np.abs(frames)))

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

        return result


def main() -> None:


    seg = Segmentation.from_basic(
        sample_rate=16000
    )