import sys
import os
from evaluate import main as evaluate_main
import librosa
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Load the test wave (normally done in data_processing.py)

test_wav_dir = os.path.join(script_dir, '../data/test-wav')
processed_relative_dir = '../data/test-processed'
processed_dir = os.path.join(script_dir, processed_relative_dir)
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

sr = 16000
n_mels = 128
hop_length = 512

for file_path in os.listdir(test_wav_dir):
    if not file_path.endswith('.wav'):
        continue
    processed_path = os.path.join(processed_dir, file_path.replace('.wav', '.npy'))
    file_path = os.path.join(test_wav_dir, file_path)
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    np.save(processed_path, mel_spec_db)


if __name__ == "__main__":
    evaluate_main(data_dir=processed_relative_dir, labels_file=None, batch_size=1, test_size=1.)