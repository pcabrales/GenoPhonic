import os
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import GPDataset
from model import GPClassifier
from utils import set_seed

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main(seed = 42,
    sr = 16000,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    hop_length = 512,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    window_size = 2,
    overlap = 1):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    data_dir = os.path.join(script_dir, '../data/processed')
    labels_file = os.path.join(script_dir, '../data/labels.csv')
    len_dataset = len(os.listdir(data_dir))
    _, test_indices = train_test_split(range(len_dataset), test_size=0.2)
    test_dataset = GPDataset(data_dir=data_dir, 
                              labels_file=labels_file, 
                              window_size=window_size,
                              overlap=overlap,
                              sr=sr,
                              hop_length=hop_length,
                              indices=test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GPClassifier(input_dim=128, num_classes=2).to(device)
    model_path = os.path.join(script_dir, '../models/model.pth')
    model.load_state_dict(torch.load(model_path))
    accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()