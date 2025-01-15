import queue
import sys
from typing import List, Literal, Self
from dataclasses import dataclass, field

import sounddevice as sd
import numpy as np

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

    def write_to_wave(self) -> None:
        ...

    def main(self) -> None:
        """The main loop where hit-to-talk happens"""
        try:
            while True:
                try:
                    input("Press any key to start recording")
                except KeyboardInterrupt:
                    print("CTRL-C also starts recording")
                
                # Routine

                pass 
        except KeyboardInterrupt:
            print("Gracefully exit")
        ...

    def routine(self) -> None:
        """Actually do things here"""
        
        ...

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