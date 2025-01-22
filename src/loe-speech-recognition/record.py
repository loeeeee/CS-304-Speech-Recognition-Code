import queue
import sys
import time
import logging
from typing import List
import os
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)
logging.basicConfig(filename='runtime.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_standard_stream(data_flow: queue.Queue, samplerate: int = 44100, channels: List = [1], dtype = np.int16) -> sd.InputStream:
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
        dtype=dtype,
        ) # Specify the 16-bit data type


def main() -> None:
    # All the default settings
    samplerate = 16000
    dtype = np.int16

    # A blocking statement
    input("Press any key to start recording!")
    time.sleep(0.1)
    print("Please start speaking at normal volume")
    
    q = queue.Queue()
    stream = create_standard_stream(q, samplerate=samplerate, dtype=dtype)

    voice_samples: np.ndarray = np.empty(1, dtype=dtype)
    try:
        with stream:
            while True:
                time.sleep(1)
                voice_samples = np.concatenate((voice_samples, q.get().reshape(-1)))
    except KeyboardInterrupt:
        print("Keyboard Interrupt received, stopping")
        try:
            while True:
                voice_samples = np.concatenate((voice_samples, q.get_nowait().reshape(-1)))
        except queue.Empty:
            print("Finish recording")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join("recordings", f"{timestamp}.wav")
    with wave.open(path, "wb") as wav:
        wav.setframerate(samplerate)
        wav.setnchannels(1)
        wav.setsampwidth(2) # 16 bit
        wav.writeframes(voice_samples.tobytes())

    return

if __name__ == "__main__":
    main()