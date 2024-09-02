import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import GPDataset
from model import ASTModel
from utils import set_seed, convert_to_list, plot_predicted_windows

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    energy_threshold = 3  # values above this are for class 1, below -energy_threshold are for class 0 and in between are for class 2 (uncertain)
    incorrect_files = {}
    additional_info_complete = {'file': [], 'time_start': [], 'time_end': [], 'predicted': [], 'probability': []}
    with torch.no_grad():
        for inputs, labels, additional_info in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs).float()
            
            # # Determining the label with softmax probabilities
            # probability = torch.sigmoid(outputs)
            # predicted = (probability > 0.5).float()
            
            # Determining the label with energy threshold
            predicted = torch.where(outputs > energy_threshold, torch.tensor(1).to(device), torch.where(outputs < -energy_threshold, torch.tensor(0).to(device), torch.tensor(2).to(device)))
            
            # # Let's find silent windows
            # for input_feature in inputs:
            #     rms = torch.mean(input_feature).item()
            #     # # Determine the label
            #     # if rms < silence_threshold:
            #     #     label = 2  # Label for silence
                
                
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # if incorrectly predicted, add the audio file name to a dict, adding one to the count if it's already there
            for i, _ in enumerate(predicted):
                if predicted[i] != labels[i]:
                    audio_file = additional_info['file'][i]
                    if audio_file in incorrect_files:
                        incorrect_files[audio_file] += 1
                    else:
                        incorrect_files[audio_file] = 1
            
            print(f'For file {additional_info["file"][0]:s}, with window starting at {additional_info["time_start"].item():.1f}s and ending at {additional_info["time_end"].item():.1f}s:')
            print(f'Predicted: {predicted.squeeze().item()}, with output value: {outputs.squeeze().item():.2f}')
            
            predicted_complete = additional_info_complete['predicted'] + convert_to_list(predicted.squeeze(1))
            additional_info_complete = {key: additional_info_complete[key] + convert_to_list(additional_info[key]) for key in additional_info}
            additional_info_complete['predicted'] = predicted_complete
            
    accuracy = 100 * correct / total
    
    return accuracy, incorrect_files, additional_info_complete

def main(
    data_dir = '../data/processed',
    labels_file = '../data/labels.csv',
    seed = 42,
    sr = 16000,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    hop_length = 512,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    window_size = 3,
    overlap = 0.5,
    batch_size = 32,
    test_size=0.2):  # Has to be 1 - train_size if testing on the same dataset as training
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    data_dir = os.path.join(script_dir, data_dir)
    len_dataset = len(os.listdir(data_dir))
    if test_size == 1.:
        test_indices = range(len_dataset)
    else:
        _, test_indices = train_test_split(range(len_dataset), test_size=test_size)
    
    test_dataset = GPDataset(data_dir=data_dir, 
                              labels_file=labels_file, 
                              window_size=window_size,
                              overlap=overlap,
                              sr=sr,
                              hop_length=hop_length,
                              indices=test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_fdim = 128  # HAS TO BE CONSISTENT WITH n_mels IN THE data_processing.py FILE
    input_tdim = test_dataset.frames_per_window

    model = ASTModel(input_tdim=input_tdim,
                     label_dim=1,
                     input_fdim=input_fdim,
                     imagenet_pretrain=False,
                     audioset_pretrain=False).to(device)
    
    model_path = os.path.join(script_dir, '../models/model.pth')
    model.load_state_dict(torch.load(model_path))
    accuracy, incorrect_files, additional_info_complete = evaluate(model, test_loader, device)
    if accuracy > 0:
        print(f'Test Accuracy: {accuracy:.2f}%')
        
    # Count the number of predictions for each class
    class_counts = {0: 0, 1: 0, 2: 0}
    for prediction in additional_info_complete['predicted']:
        class_counts[prediction] += 1
    print(f'Class counts: {class_counts}')
    print(f'Class 0 spoke {class_counts[0] / (class_counts[1] + class_counts[0]) * 100:.2f} % of times')
    
    # # order the dict by counts
    # incorrect_files = {k: v for k, v in sorted(incorrect_files.items(), key=lambda item: item[1], reverse=True)}
    # print(f'Incorrectly predicted files: {incorrect_files}')
    # print(f'Number of incorrectly predicted files: {len(incorrect_files)}') 
    
    save_plot_dir = os.path.join(script_dir, '../images/test-predicted.png')
    label_names={0: 'Pablo', 1: 'Ginebra', 2:'Silence/Uncertain'}
    plot_predicted_windows(additional_info_complete, save_plot_dir, label_names=label_names)
    
    
if __name__ == "__main__":
    main()