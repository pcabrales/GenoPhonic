import sys
import subprocess
import os
from evaluate import main as evaluate_main
import librosa
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Load the test wave (normally done in data_processing.py)
base_dir = 'conversaciones-gp-agosto-2024'
test_wav_dir = os.path.join(script_dir, f'../data/{base_dir}-wav')

# Create the processed directory if it doesn't exist
if not os.path.exists(test_wav_dir):
    os.makedirs(test_wav_dir)
    test_dir = test_wav_dir[:-4]
    if not os.path.exists(test_dir):
        raise FileNotFoundError('Test directory not found')
    for file_name in os.listdir(test_dir):
        input_path = os.path.join(test_dir, file_name)
        if file_name.endswith('.opus'):
            output_path = os.path.join(test_wav_dir, file_name.replace('.opus', '.wav'))
        elif file_name.endswith('.ogg'):
            output_path = os.path.join(test_wav_dir, file_name.replace('.ogg', '.wav'))
        subprocess.run(['ffmpeg', '-i', input_path, output_path])

processed_relative_dir = f'../data/{base_dir}-processed'
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