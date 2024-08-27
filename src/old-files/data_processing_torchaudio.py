import os
import torchaudio
import librosa
import numpy as np
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

wav_dir = os.path.join(script_dir, '../data/wav')
processed_dir = os.path.join(script_dir, '../data/processed_torchaudio')
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
    y, sr = torchaudio.load(file_path)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, hop_length=hop_length)
    y = y.unsqueeze(0)
    mel_spec = mel_transform(y)
    mel_spec = mel_spec.unsqueeze(0)
    
    np.save(processed_path, mel_spec)