import os
import csv
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Define the input and output file paths
input_file_path = os.path.join(script_dir, '../data/_chat.txt')
output_file_path = os.path.join(script_dir, '../data/labels.csv')

# Open the input and output files
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header row
    csv_writer.writerow(['file_path', 'label'])
    for line in infile:
        if '.opus' in line:
            # Find the starting index of the file name
            start_index = line.rfind(' ', 0, line.find('.opus')) + 1
            # Extract the audio file name
            audio_file_name = line[start_index:line.find('.opus')]
            # Determine the sender and assign the corresponding value
            if 'Pablo' in line:
                label = 0
            elif 'gine' in line:
                label = 1
            else:
                continue  # Skip lines that don't match either sender
            print(audio_file_name, label)
            # Write the output line to the new file
            csv_writer.writerow([audio_file_name, label])

# 533 866 audios mios vs de gine