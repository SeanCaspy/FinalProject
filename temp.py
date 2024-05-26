import soundfile as sf
import numpy as np
import scipy.signal


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def apply_noise_reduction(y, sr):
    cutoff_freq = 2000
    b, a = scipy.signal.butter(4, cutoff_freq / (0.5 * sr), 'low')
    filtered_y = scipy.signal.lfilter(b, a, y)
    return filtered_y


def save_filtered_audio(filtered_y, sr, output_path):
    sf.write(output_path, filtered_y, sr)


if __name__ == "__main__":
    input_file = '/Users/Sean/Desktop/record.wav'
    output_file = '/Users/Sean/Desktop/filtered_output.wav'
    y, sr = load_audio(input_file)
    filtered_y = apply_noise_reduction(y, sr)
    save_filtered_audio(filtered_y, sr, output_file)

