import os
import librosa
import numpy as np
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

wav_dir = os.path.join(script_dir, '../data/wav')
processed_dir = os.path.join(script_dir, '../data/processed')
# Create the processed directory if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

sr = 16000
n_mels = 128
hop_length = 512

for file_path in os.listdir(wav_dir):
    if not file_path.endswith('.wav'):
        continue
    processed_path = os.path.join(processed_dir, file_path.replace('.wav', '.npy'))
    file_path = os.path.join(wav_dir, file_path)
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features = mel_spec_db.T  # Transpose to make it (time, features)
    np.save(processed_path, features)