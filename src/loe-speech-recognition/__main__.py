
import queue
import sys
import time
import wave

import numpy as np
import sounddevice as sd

def main() -> None:
    # All the default settings
    samplerate = 16000
    channels = [1]
    mapping = [c - 1 for c in channels]  # Channel numbers start with 1

    q = queue.Queue()
    # Def callback action
    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        q.put(indata[::1, mapping])

    stream = sd.InputStream(
        channels=max(channels),
        samplerate=samplerate, 
        callback=audio_callback,
        dtype=np.int16,
        ) # Specify the 16-bit data type

    # A blocking statement
    input("Press any key to start recording!")

    start_time = time.time()
    timeout = 5
    # The recording starts
    with wave.open("test.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setframerate(samplerate)
        wav.setsampwidth(2) # Meaning 16 bit 
        with stream:
            try:
                accumulated_silent_time: float = 0.0
                silent_time_threshold: float = 0.2
                background_noise_threshold: int = 20
                while time.time() - start_time < timeout:
                    frame: np.ndarray = q.get().flatten()
                    wav.writeframes(frame.tobytes())
            except KeyboardInterrupt:
                print("Keyboard interrupt received, stopping")
                pass

if __name__ == "__main__":
    main()
