import argparse
from train import main as test_main

# Load the test wave (normally done in data_processing.py)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the GenoPhonic model to identify audio speakers")
    parser.add_argument('--audio_dir', type=str, default='../data/conversaciones-gp-agosto-2024', help='Directory containing audio files')
    args = parser.parse_args()
    test_main(audio_dir=args.audio_dir, labels_file=None, batch_size=1, test_size=1., only_evaluate=True)

# To run from the command line:
# python test.py
# Or:
# python train.py --audio_dir ../data/conversaciones-gp-agosto-2024 --batch_size 1 --test_size 1. --only_evaluate