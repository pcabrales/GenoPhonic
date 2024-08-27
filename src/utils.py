import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt 
import matplotlib

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

def plot_predicted_windows(data_dict, save_plot_dir, label_names={0: 'Pablo', 1: 'Ginebra', 2:'Silence/Uncertain'}):
    # Enable grid
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get unique filenames to plot them separately
    unique_files = sorted(set(data_dict['file']))

    # Define colors for labels
    colors = {0: 'red', 1: 'green', 2: 'gray'}

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
    # red_patch = matplotlib.patches.Patch(color='red', label='Pablo')
    # green_patch = matplotlib.patches.Patch(color='green', label='Ginebra')
    handles = [matplotlib.patches.Patch(color=colors[label], label=label_names[label]) for label in colors]

    # Add legend to plot
    plt.legend(handles=handles, title='Labels', loc='lower right')
    plt.grid(True, linewidth=2.5)
    
    plt.savefig(save_plot_dir)