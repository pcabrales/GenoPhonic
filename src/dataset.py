import os 
import csv
import torch
from torch.utils.data import Dataset

class GenoPhonicDataset(Dataset):
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        with open(labels_file, mode='r') as file:
            csv_reader = csv.reader(file)
        