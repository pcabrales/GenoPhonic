import os
import sys
import subprocess
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

input_folder = os.path.join(script_dir, '../data/raw')
output_folder =  os.path.join(script_dir, '../data/wav')
# Create the processed directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.opus'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace('.opus', '.wav'))
        subprocess.run(['ffmpeg', '-i', input_path, output_path])