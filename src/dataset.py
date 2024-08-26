import os 
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class GPDataset(Dataset):
    
    # using a sliding window approach to divide the audio file into smaller segments
    
    def __init__(self, data_dir, window_size, overlap, sr, hop_length, indices=None, labels_file=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.sr = sr
        self.hop_length = hop_length
        
        self.frames_per_window = int(self.window_size * sr / hop_length)
        self.frames_per_overlap = int(self.overlap * sr / hop_length)
        self.step_size = self.frames_per_window - self.frames_per_overlap
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'{data_dir} does not exist')
        
        if labels_file is not None:
            self.df_labels = pd.read_csv(labels_file)
        else:
            self.df_labels = pd.DataFrame({'file': [file.replace('.npy', '') for file in os.listdir(data_dir)]})
            self.df_labels['label'] = np.nan    
        
        if indices is not None:
            self.df_labels = self.df_labels.iloc[indices].reset_index(drop=True)
            
        # Preprocess dataset to map each index to a file and window
        self.index_map = []
        for file_idx in range(len(self.df_labels)):
            file_path = os.path.join(self.data_dir, self.df_labels.iloc[file_idx, 0] + '.npy')
            features = np.load(file_path)
            num_frames = features.shape[1]  # Number of time frames
            num_windows = (num_frames - self.frames_per_overlap) // self.step_size
            
            for window_idx in range(num_windows):
                self.index_map.append((file_idx, window_idx))
            
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        file_idx, window_idx = self.index_map[idx]
        file_path = os.path.join(self.data_dir, self.df_labels.iloc[file_idx, 0] + '.npy')
        features = np.load(file_path)
        label = self.df_labels.iloc[file_idx, 1]
        
        window_start = window_idx * self.step_size
        window_end = window_start + self.frames_per_window
        window_features = torch.tensor(features[:, window_start:window_end])
        
        additional_info = {'file': self.df_labels.iloc[file_idx, 0],
                           'time_start': window_start * self.hop_length / self.sr,
                           'time_end': window_end * self.hop_length / self.sr
                           }

        return window_features, label, additional_info
    