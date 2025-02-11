import queue
import sys
import time
from typing import List

import numpy as np
import sounddevice as sd


def create_standard_stream(data_flow: queue.Queue, samplerate: int = 44100, channels: List = [1]) -> sd.InputStream:
    mapping = [c - 1 for c in channels]  # Channel numbers start with 1

    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        data_flow.put(indata[::1, mapping])
    
    return sd.InputStream(
        channels=max(channels),
        samplerate=samplerate, 
        callback=audio_callback,
        dtype=np.int16,
        ) # Specify the 16-bit data type


def main() -> None:
    # All the default settings
    samplerate = 16000

    q = queue.Queue()
    stream = create_standard_stream(q, samplerate=samplerate)

    print("Testing noise floor")
    time.sleep(1)
    with stream:
        time.sleep(3)
        background_samples: np.ndarray = q.get().reshape(-1)
        try:
            while True:
                background_samples = np.concatenate((background_samples, q.get_nowait().reshape(-1)))
        except queue.Empty:
            noise_floor = int(np.average(np.abs(background_samples)))
            print(noise_floor)

    # A blocking statement
    input("Press any key to start recording!")
    time.sleep(1)
    print("Please start speaking at normal volume")
    
    q = queue.Queue()
    stream = create_standard_stream(q, samplerate=samplerate)

    with stream:
        time.sleep(3)
        voice_samples: np.ndarray = q.get().reshape(-1)
        try:
            while True:
                voice_samples = np.concatenate((voice_samples, q.get_nowait().reshape(-1)))
        except queue.Empty:
            speech_threshold = int(np.average(np.abs(voice_samples) - noise_floor))
            print("Finish testing")
            print(speech_threshold)

    return

if __name__ == "__main__":
    main()