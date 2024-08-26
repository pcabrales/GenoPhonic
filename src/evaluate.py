import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import GPDataset
from model import ASTModel
from utils import set_seed, convert_to_list

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    incorrect_files = {}
    additional_info_complete = {'file': [], 'time_start': [], 'time_end': [], 'predicted': []}
    with torch.no_grad():
        for inputs, labels, additional_info in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs).float()
            predicted = (torch.nn.functional.sigmoid(outputs) > 0.5).float()  # for binary classification
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
            print(f'Predicted: {predicted.squeeze().item()}')
            
            predicted_complete = additional_info_complete['predicted'] + convert_to_list(predicted.squeeze(1))
            additional_info_complete = {key: additional_info_complete[key] + convert_to_list(additional_info[key]) for key in additional_info}
            additional_info_complete['predicted'] = predicted_complete
            
    print(additional_info_complete)
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
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # # order the dict by counts
    # incorrect_files = {k: v for k, v in sorted(incorrect_files.items(), key=lambda item: item[1], reverse=True)}
    # print(f'Incorrectly predicted files: {incorrect_files}')
    # print(f'Number of incorrectly predicted files: {len(incorrect_files)}') 
    
    ###
    import matplotlib.pyplot as plt 

    # Enable grid
    # Create a figure and axis
    data_dict = additional_info_complete
    fig, ax = plt.subplots()

    # Get unique filenames to plot them separately
    unique_files = sorted(set(data_dict['file']))

    # Define colors for labels
    colors = {0: 'red', 1: 'green'}

    # Iterate over unique filenames
    for file in unique_files:
        # Get indices where filename matches
        indices = [i for i, fname in enumerate(data_dict['file']) if fname == file]
        
        # Plot each window for this file
        for idx in indices:
            start_time = data_dict['time_start'][idx]
            end_time = data_dict['time_end'][idx]
            label = data_dict['predicted'][idx]
            
            # Draw a horizontal line for each window
            ax.hlines(y=file, xmin=start_time, xmax=end_time, color=colors[label], linewidth=5)

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('File')
    ax.set_title('Voice Classification GenoPhonic (GP)')

    ax.set_yticks(range(len(unique_files)))
    ax.set_yticklabels(unique_files)

    # Adjust y-axis label rotation and spacing
    plt.yticks(rotation=0)  # Rotate y-axis labels for better fit
    plt.tight_layout(pad=4)  # Adjust layout to make space for y-axis labels

    # Increase space on the left to fit the labels
    plt.subplots_adjust(left=0.4)
    
    import matplotlib
    
    # Create custom legend
    red_patch = matplotlib.patches.Patch(color='red', label='Pablo')
    green_patch = matplotlib.patches.Patch(color='green', label='Ginebra')

    # Add legend to plot
    plt.legend(handles=[red_patch, green_patch], title='Labels', loc='upper right')
    plt.grid(True, linewidth=2.5)
    
    plt.savefig(os.path.join(script_dir, '../images/test-predicted.png'))
    ###
    
if __name__ == "__main__":
    main()