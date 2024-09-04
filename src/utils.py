import os
import subprocess
import csv
from tqdm import tqdm
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import librosa

def set_seed(seed):
    """
    Set all the random seeds to a fixed value to take out any randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True

def convert_to_list(value):
    if torch.is_tensor(value):
        return value.tolist()
    return value

def plot_predicted_windows(data_dict, save_plot_dir, label_names={0: 'Pablo', 1: 'Ginebra', -1:'Silence/Uncertain'}):
    # Enable grid
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get unique filenames to plot them separately
    unique_files = sorted(set(data_dict['file']))

    # Define colors for labels
    colors = {-1: 'gray', 0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'orange', 5: 'cyan'}

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
    
    # Create custom legend
    handles = [matplotlib.patches.Patch(color=colors[label], label=label_names[label]) for label in label_names]

    # Add legend to plot
    plt.legend(handles=handles, title='Labels', loc='lower right')
    plt.grid(True, linewidth=2.5)
    
    plt.savefig(save_plot_dir)


def label_dataset(chat_file, labels_file, speakernames):
    # if labels_file exists, skip the labeling process
    if os.path.exists(labels_file):
        print('Labels file already exists')
        return
    file_types = ['.ogg', '.opus']
    # Open the input and output files
    with open(chat_file, 'r', encoding='utf-8') as infile, open(labels_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(['file_path', 'label'])
        for line in infile:
            for file_type in file_types:
                if file_type in line:
                    # Find the starting index of the file name
                    start_index = line.rfind(' ', 0, line.find('.opus')) + 1
                    # Extract the audio file name
                    audio_file_name = line[start_index:line.find('.opus')]
                    # Determine the sender and assign the corresponding value
                    for speakername in speakernames:
                        if speakername in line:
                            label = speakernames.index(speakername)
                            # Write the output line to the new file
                            csv_writer.writerow([audio_file_name, label])
                            break
                    
                    break
    return
# 533 866 audios mios vs de gine


def convert_to_wav(input_folder):
    if not os.path.exists(input_folder):
        raise FileNotFoundError('Data folder not found')
    output_folder = input_folder + '-wav'
    # if the output folder exists, skip the conversion process
    if os.path.exists(output_folder):
        print('Wav files already exist')
        return
    # Create the processed directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Converting to wav')
    for file_name in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file_name)
        if file_name.endswith('.opus'):
            output_path = os.path.join(output_folder, file_name.replace('.opus', '.wav'))
        elif file_name.endswith('.ogg'):
            output_path = os.path.join(output_folder, file_name.replace('.ogg', '.wav'))
        subprocess.run(['ffmpeg', '-i', input_path, '-loglevel', 'quiet', output_path])
    return


def convert_to_mel_spectrogram(input_folder, sr=16000, n_mels=128, hop_length=512):
    output_folder = input_folder + '-processed'
    if os.path.exists(output_folder):
        print('Mel spectrograms already exist')
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    wav_dir = input_folder + '-wav'
    print('Converting to mel spectrogram')
    for file_path in tqdm(os.listdir(wav_dir)):
        if not file_path.endswith('.wav'):
            continue
        processed_path = os.path.join(output_folder, file_path.replace('.wav', '.npy'))
        file_path = os.path.join(wav_dir, file_path)
        y, sr = librosa.load(file_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        np.save(processed_path, mel_spec_db)
    return