import os
import sys
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from dataset import GPDataset
# from model import GPClassifier
from model import ASTModel
from utils import set_seed, plot_predicted_and_spectrogram, label_dataset, convert_to_wav, convert_to_mel_spectrogram, convert_to_list, plot_predicted_windows

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
        inputs = inputs.permute(0, 2, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs).float()
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # _, predicted = torch.max(outputs, 1)  # for multi-class classification
        predicted = (torch.nn.functional.sigmoid(outputs) > 0.5).float()  # for binary classification
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return running_loss / len(train_loader), accuracy

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    energy_threshold = 3  ### values above this are for class 1, below -energy_threshold are for class 0 and in between are for class 2 (uncertain)
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
            predicted = torch.where(outputs > energy_threshold, torch.tensor(1).to(device), torch.where(outputs < -energy_threshold, torch.tensor(0).to(device), torch.tensor(-1).to(device)))
            
            # # Let's find silent windows
            # for input_feature in inputs:
            #     rms = torch.mean(input_feature).item()
            #     # # Determine the label
            #     # if rms < silence_threshold:
            #     #     label = -1  # Label for silence
            
            # if incorrectly predicted, add the audio file name to a dict, adding one to the count if it's already there
            for i, _ in enumerate(predicted):
                if predicted[i] != labels[i]:
                    audio_file = additional_info['file'][i]
                    if audio_file in incorrect_files:
                        incorrect_files[audio_file] += 1
                    else:
                        incorrect_files[audio_file] = 1
            
            # print(f'For file {additional_info["file"][0]:s}, with window starting at {additional_info["time_start"].item():.1f}s and ending at {additional_info["time_end"].item():.1f}s:')
            # print(f'Predicted: {predicted.squeeze().item()}, with output value: {outputs.squeeze().item():.2f}')
            predicted_complete = additional_info_complete['predicted'] + convert_to_list(predicted.squeeze(1))
            additional_info_complete = {key: additional_info_complete[key] + convert_to_list(additional_info[key]) for key in additional_info}
            additional_info_complete['predicted'] = predicted_complete
            
            # only keep predicted values that are not -1
            labels = labels[predicted != -1]
            predicted = predicted[predicted != -1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    
    return accuracy, incorrect_files, additional_info_complete

def main(
    audio_dir='../data/GinePablo',
    labels_file='../data/labels.csv',
    speakernames=['Pablo', 'gine'],
    model_path = '../models/model.pth',
    seed=42,
    sr=16000,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    hop_length=512,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    window_size=3,
    n_mels=128,
    overlap=0.5,
    N_epochs=10,
    batch_size = 32,
    test_size=0.2,  # Has to be 1 - train_size if testing on the same dataset as training
    only_evaluate=False):
    
    set_seed(seed)
    dataset_name = os.path.basename(audio_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    audio_dir = os.path.join(script_dir, audio_dir)
    chat_file = os.path.join(audio_dir, '_chat.txt')
    if labels_file is not None:
        labels_file = os.path.join(script_dir, labels_file)
        # Label the dataset
        label_dataset(chat_file, labels_file, speakernames)
    model_path = os.path.join(script_dir, model_path)
    # Convert the dataset to wav
    convert_to_wav(audio_dir)
    # Convert the wav files to mel spectrograms
    convert_to_mel_spectrogram(audio_dir, sr=sr, n_mels=128, hop_length=hop_length)
    processed_dir = audio_dir + '-processed'
    len_dataset = len(os.listdir(processed_dir))
    if test_size == 1.:
        test_indices = range(len_dataset)
    else:
        train_indices, test_indices = train_test_split(range(len_dataset), test_size=test_size)
    
    # Test dataset
    test_dataset = GPDataset(data_dir=processed_dir, 
                              labels_file=labels_file, 
                              window_size=window_size,
                              overlap=overlap,
                              sr=sr,
                              hop_length=hop_length,
                              indices=test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_tdim = test_dataset.frames_per_window
    model = ASTModel(input_tdim=input_tdim,
                    label_dim=1,
                    input_fdim=n_mels,
                    imagenet_pretrain=False,
                    audioset_pretrain=False).to(device)
    
    if not only_evaluate:
        train_dataset = GPDataset(data_dir=processed_dir,
                                labels_file=labels_file,
                                window_size=window_size,
                                overlap=overlap,
                                sr=sr,
                                hop_length=hop_length,
                                indices=train_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_tdim = train_dataset.frames_per_window
        criterion = BCEWithLogitsLoss()
        
        ## ADAMW
        optimizer = AdamW(model.parameters(), lr=0.0001)
        
        ## ADAM
        # trainables = [p for p in model.parameters() if p.requires_grad]
        # print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
        # print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        # optimizer = torch.optim.Adam(trainables, 0.0001, weight_decay=5e-7, betas=(0.95, 0.999))

        for epoch in range(N_epochs):  # Define your number of epochs
            loss, accuracy = train(model, train_loader, criterion, optimizer, device)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        torch.save(model.state_dict(), model_path)

    # Evaluate the model
    model.load_state_dict(torch.load(model_path))
    accuracy, incorrect_files, additional_info_complete = evaluate(model, test_loader, device)
    if accuracy > 0:
        print(f'Test Accuracy: {accuracy:.2f}%')
        
    # Count the number of predictions for each class
    class_counts = {speakernames.index(speakername): 0 for speakername in speakernames}
    class_counts[-1] = 0
    for prediction in additional_info_complete['predicted']:
        class_counts[prediction] += 1
    
    speakername_counts = {speakername: class_counts[speakernames.index(speakername)] for speakername in speakernames}
    print(f'speakername counts: {speakername_counts}')
    total_speakername_counts = sum(speakername_counts.values())
    for speakername in speakernames:
        print(f'{speakername:s} spoke {speakername_counts[speakername] / total_speakername_counts * 100:.2f} % of times')
    
    label_names = {speakernames.index(speakername): speakername for speakername in speakernames}
    label_names[-1] = 'Silence/Uncertain'
    # order the dict by counts
    if labels_file is not None:
        incorrect_files = {k: v for k, v in sorted(incorrect_files.items(), key=lambda item: item[1], reverse=True)}
        print('Incorrectly predicted files (ten first values):')
        print({k: incorrect_files[k] for k in list(incorrect_files)[:10]})
        print(f'Number of incorrectly predicted files: {len(incorrect_files)}')
    # if labels_file is None:
    #     save_plot_dir = os.path.join(script_dir, f'../images/{dataset_name}-spectrogram-predicted.png')
    #     file_path = 'mix-voice-pablo-phone'
    #     plot_predicted_and_spectrogram(file_path, processed_dir, additional_info_complete, save_plot_dir, label_names=label_names)
        
    
    save_plot_dir = os.path.join(script_dir, f'../images/{dataset_name}-predicted.png')
    plot_predicted_windows(additional_info_complete, save_plot_dir, label_names=label_names)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the GenoPhonic model to identify audio speakers")
    parser.add_argument('--audio_dir', type=str, default='../data/GinePablo', help='Directory containing audio files')
    parser.add_argument('--labels_file', type=str, default='../data/labels.csv', help='Labels file path')
    parser.add_argument('--speakernames', nargs='+', default=['Pablo', 'gine'], help='List of speakers')
    parser.add_argument('--model_path', type=str, default='../models/model.pth', help='Path to save/load the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for audio processing')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for audio processing')
    parser.add_argument('--window_size', type=int, default=3, help='Window size for audio segmentation')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands for spectrograms')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap for window segmentation')
    parser.add_argument('--N_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of dataset to include in test split')
    parser.add_argument('--only_evaluate', action='store_true', help='Only evaluate the model without training')
    
    args = parser.parse_args()
    
    main(
        audio_dir=args.audio_dir,
        labels_file=args.labels_file,
        speakernames=args.speakernames,
        model_path=args.model_path,
        seed=args.seed,
        sr=args.sr,
        hop_length=args.hop_length,
        window_size=args.window_size,
        n_mels=args.n_mels,
        overlap=args.overlap,
        N_epochs=args.N_epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        only_evaluate=args.only_evaluate
    )
