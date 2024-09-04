# GenoPhonic (GP) for AI Voice ID 🎤🤖〰️

GP is an AI tool that can tell if a voice in an audio recording belongs to one person or another. It’s designed to distinguish between two specific people (like you and someone else) by analyzing the unique features of their voices.

## What It Does:
- **Voice Identification**: It listens to a voice and figures out which person it belongs to, based on what it’s been trained on.
- **AI-Powered**: Uses machine learning to get better at recognizing the voices over time.
- **Easy to Train**: You can quickly train it to recognize your voice and another person's voice.

## How You Might Use It:
- For personal use, like setting up voice-based security for your devices.
- To make your smart home respond differently depending on who's talking.
- Or just for fun, to see how well it can tell voices apart.


## Getting Started:
- Clone the repo.
```bash
git clone https://github.com/pcabrales/GenoPhonic.git
```

- Follow the instructions to train the AI with your own voice samples.
- Start identifying who’s talking in your recordings!

## Setup Instructions:

### 1. Create a Conda Environment

First, create a new Conda environment to manage the dependencies for this project called `gp` from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate gp
```

### 2. Install FFmpeg
This project requires FFmpeg to convert OPUS files to WAV format. Please install FFmpeg according to your operating system:

- **Windows**: Download the latest build from [here](https://ffmpeg.org/download.html#build-windows).
- **macOS**: Install via Homebrew: `brew install ffmpeg`.
- **Linux**: Install via package manager: `sudo apt-get install ffmpeg`.

### 3. Data Preparation
- **IPhone**: 
    - **Training**: In WhatsApp, `Export Chat` for the conversation that include at least tens to hundreds of voice messages for each speaker you wish to ID. Extract it to the `data` folder and run.
    - **Testing**: Record as many voice messages as desired of conversations between the two speakers you wish to ID. Save it in a folder in the `data` folder.

### 4. Train the Model
Run the following command to train the model with the provided data:

```bash
python src/train.py  --audio_dir path/to/audio_dir_training --labels_name ['speaker1', 'speaker2']
```
There are other optional arguments you can use to customize the training process. Run `python src/train.py --help` to see them.

You will see the accuracy of the model on the training and validation sets. The model will be saved in the `models` folder.

### 5. Test the Model
Run the following command to test the model with the provided data:

```bash
python src/test.py  --audio_dir path/to/audio_dir_testing'
```
You will see how much each speaker spoke relative to the total, and images with visualizations of the model's predictions will be saved.


## License:
This project is under a GNU Affero General Public License v3.0
