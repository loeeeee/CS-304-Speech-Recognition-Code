
import queue
import sys
import time
import wave

import numpy as np
import sounddevice as sd

# Main Loop

## Don't stop until CTRL-D is received

## Start record when recording is hit


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
        wav.setframerate(16000)
        wav.setsampwidth(2)
        with stream:
            try:
                while time.time() - start_time < timeout:
                    frame = q.get()
                    wav.writeframes(frame.flatten().tobytes())
            except KeyboardInterrupt:
                print("Keyboard interrupt received, stopping")
                pass

if __name__ == "__main__":
    main()
