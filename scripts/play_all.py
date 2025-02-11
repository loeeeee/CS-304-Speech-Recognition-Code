import os
import sys
import time

import sounddevice as sd
import numpy as np
import soundfile as sf

start_idx = 0
def play_sine_wave() -> None:
    samplerate = 16000
    amplitude = 0.08
    frequency = 500
    
    def callback(outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        global start_idx
        t = (start_idx + np.arange(frames)) / samplerate
        t = t.reshape(-1, 1)
        outdata[:] = amplitude * np.sin(2 * np.pi * frequency * t)
        start_idx += frames

    with sd.OutputStream(channels=1, callback=callback,
                         samplerate=samplerate):
        time.sleep(0.3)

def main() -> None:
    path = "segment_results"
    for file_name in os.listdir(path):
        full_path = os.path.join(path, file_name)
        data, fs = sf.read(full_path, dtype='float32')
        data = data # * 1.5
        sd.play(data, fs)
        sd.wait()
        play_sine_wave()

if __name__ == "__main__":
    main()