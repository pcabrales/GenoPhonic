import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from sklearn.model_selection import train_test_split
from dataset import GPDataset
# from model import GPClassifier
from model import ASTModel
from utils import set_seed

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
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

def main(seed = 42,
    sr = 16000,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    hop_length = 512,  # HAS TO BE CONSISTENT WITH THE data_processing.py FILE
    window_size = 2,
    overlap = 1,
    N_epochs = 2):
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    data_dir = os.path.join(script_dir, '../data/processed')
    labels_file = os.path.join(script_dir, '../data/labels.csv')
    len_dataset = len(os.listdir(data_dir))
    train_indices, _ = train_test_split(range(len_dataset), test_size=0.2)
    train_dataset = GPDataset(data_dir=data_dir, 
                              labels_file=labels_file, 
                              window_size=window_size,
                              overlap=overlap,
                              sr=sr,
                              hop_length=hop_length,
                              indices=train_indices)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_fdim = 128  # HAS TO BE CONSISTENT WITH n_mels IN THE data_processing.py FILE
    input_tdim = train_dataset.frames_per_window
    print(input_tdim)
    model = ASTModel(input_tdim=input_tdim,
                     label_dim=1,
                     input_fdim=input_fdim,
                     imagenet_pretrain=False,
                     audioset_pretrain=False).to(device)
    criterion = BCEWithLogitsLoss()
    # optimizer = AdamW(model.parameters(), lr=0.0001)
    
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, 0.001, weight_decay=5e-7, betas=(0.95, 0.999))

    for epoch in range(N_epochs):  # Define your number of epochs
        loss, accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    model_path = os.path.join(script_dir, '../models/model.pth')
    torch.save(model.state_dict(), model_path)
        
if __name__ == "__main__":
    main()