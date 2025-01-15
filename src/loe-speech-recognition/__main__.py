import itertools
import queue
import sys
import time
from typing import List
import wave

import numpy as np
import sounddevice as sd

def detect_segment(signal: queue.Queue, result: List[np.ndarray], frame_size: int = 1600, high_threshold: int=512, noise_floor:int=128) -> bool:
    # Setup flags
    """
    Implements a two-threshold adaptive endpointing algorithm.

    Args:
        signal: The input audio signal as a NumPy array.
        frame_size: The size of each frame in samples.
        hop_size: The hop size between frames in samples.
        high_threshold: The high threshold for speech activity detection.
        low_threshold: The low threshold for speech activity detection.

    Returns:
        A list of indices indicating the start and end frames of speech segments.
    """
    # Get all cached signal
    audio = signal.get(timeout=0.1).reshape(-1)
    # print(f"First audio {audio}")
    try:
        while True:
            next_audio = signal.get_nowait().reshape(-1)
            # print(f"Next audio {next_audio}")
            audio = np.concatenate((audio, next_audio))
    except queue.Empty:
        pass


    def isSpeech(current_frame: np.ndarray) -> bool:
        abs_total_frame = np.abs(current_frame) - noise_floor
        average_energy = np.average(abs_total_frame)
        if average_energy > high_threshold:
            return True
        else:
            return False

    num_of_frames: int = audio.shape[0] // frame_size

    speech_detected = False
    trimmed_audio = audio[:frame_size*num_of_frames]
    # print(trimmed_audio.shape)
    cool_down_counter = 3
    for frame in itertools.chain.from_iterable((trimmed_audio.reshape((-1, frame_size)), [audio[frame_size*num_of_frames:]])):
        if isSpeech(frame):
            cool_down_counter = 3
            result.append(frame)
            speech_detected = True
        elif speech_detected:
            cool_down_counter -= 1
            result.append(frame)
            if cool_down_counter == 0:
                break

    return speech_detected

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

    # Find noise floor
    print("Initialize devices")
    time.sleep(0.2)
    with stream:
        time.sleep(0.5)
        background_samples: np.ndarray = q.get().reshape(-1)
        try:
            while True:
                background_samples = np.concatenate((background_samples, q.get_nowait().reshape(-1)))
        except queue.Empty:
            noise_floor = int(np.average(np.abs(background_samples)))

    # A blocking statement
    input("Press any key to start recording!")
    time.sleep(1) # Avoid keyboard noise
    print("Start Recording")
    
    q = queue.Queue()
    stream = create_standard_stream(q,samplerate=samplerate)
    
    start_time = time.time()
    timeout = 30
    segment_detection_interval: float = 1 # Do it every 0.2 seconds
    last_segment_detection: float = time.time() + 0.2
    results: List = []
    speech_ever_detected: bool = False
    # The recording starts
    with wave.open("test.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setframerate(samplerate)
        wav.setsampwidth(2) # Meaning 16 bit 
        with stream:
            try:
                while time.time() - start_time < timeout:
                    if last_segment_detection + segment_detection_interval <= time.time():
                        last_segment_detection = time.time()
                        detect_result = detect_segment(q, results, noise_floor=noise_floor)
                        if detect_result:
                            # Return true if speech was never detected or it is in speech
                            speech_ever_detected = True
                            continue
                        elif not detect_result and speech_ever_detected:
                            # Return false if speech was over
                            break
                        else:
                            continue
                    else:
                        # Sleep till the next detection
                        time.sleep(segment_detection_interval)
                # Get the segmented speech
                segmented_speech: np.ndarray = np.concatenate(results)
                # sd.play(segmented_speech, 16000, blocking=True)
                wav.writeframes(segmented_speech.tobytes())
            except KeyboardInterrupt:
                print("Keyboard interrupt received, stopping")
                pass

if __name__ == "__main__":
    main()
