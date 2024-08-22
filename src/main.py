import sys
from train import main as train_main
from evaluate import main as evaluate_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_main()
    else:
        print("Please provide 'train' or 'evaluate' as an argument.")
