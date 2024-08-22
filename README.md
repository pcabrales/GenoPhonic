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
- Follow the instructions to train the AI with your own voice samples.
- Start identifying who’s talking in your recordings!

## Setup Instructions:

### 1. Create a Conda Environment

First, create a new Conda environment to manage the dependencies for this project:

```bash
conda env create -f environment.yml
conda activate voice_classifier
```

### 2. Install FFmpeg
This project requires FFmpeg to convert OPUS files to WAV format. Please install FFmpeg according to your operating system:

- **Windows**: Download the latest build from [here](https://ffmpeg.org/download.html#build-windows).
- **macOS**: Install via Homebrew: `brew install ffmpeg`.
- **Linux**: Install via package manager: `sudo apt-get install ffmpeg`.

### 3. Data Preparation
This project requires FFmpeg to convert OPUS files to WAV format. Please install FFmpeg according to your operating system:

```bash
python src/data_processing.py
```
To start training, run:
```bash
python main.py train
```

To evaluate the model, run:
```bash
python main.py evaluate
```

## License:
This project is under a GNU Affero General Public License v3.0
