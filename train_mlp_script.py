import pickle
import numpy as np
import sparselinear as sl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import sparselinear as sl
import os
from model_codes import *

mlp_checkpoints_path = "/home/arpita-local/nn_structure_project/mlp_checkpoints"

if not os.path.exists(mlp_checkpoints_path): 
    os.makedirs(mlp_checkpoints_path)



def load_data():
    home = "/storage/toki-local/ECG_DATA/"
    with open(home+'output_nature_balanced','rb') as f: y1= pickle.load(f)
    with open(home+'input_nature_balanced','rb') as f: x1= pickle.load(f)

    x=x1
    y=y1
    x=x1
    y=y1
    x=np.nan_to_num(x)
    x = x[:,:,[0]]

    x_dummy= torch.tensor(x, dtype=torch.float32)
    y_dummy= torch.tensor(y, dtype=torch.long)
    x_shape= x_dummy.shape

    # Convert numpy arrays to PyTorch tensors
    x_tensor = torch.tensor(x_dummy).reshape(x_shape[0], -1)  # Flatten the input
    y_tensor = torch.tensor(y_dummy)

    return x_tensor, y_tensor

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# #initialize the network with he initialization
def init_weights(m):
    # set seed for reproducibility
    torch.manual_seed(0)

    
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
    
# initialize the network with xavier initialization
# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_normal_(m.weight)
#         m.bias.data.fill_(0.01)


# First, define the device based on the availability of CUDA
def train_mlp(model, num_epochs, ckpt_root, save=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the defined device
    #model = model.to(device)

    # Training loop
    #num_epochs = 100

    training_acc = []
    validation_acc = []
    #ckpt_root= 'structural_variation/model_1_500_200_300'
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for inputs, targets in train_loader:
            # Move inputs and targets to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs,_ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Ensure validation data is also on the correct device
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs,_ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        training_acc.append(train_acc)
        validation_acc.append(val_acc)
        

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if save==True:
            torch.save(model.state_dict(), f"{ckpt_root}/ckpt_{epoch}.pth")


    #save the train and validation accuracy
    if save==True:
        with open(f"{ckpt_root}/training_acc.pkl", "wb") as f:
            pickle.dump(training_acc, f)
        with open(f"{ckpt_root}/validation_acc.pkl", "wb") as f:
            pickle.dump(validation_acc, f)


# Create a dataset and split it into training and validation sets

x_tensor, y_tensor = load_data()
dataset = CustomDataset(x_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = MLP_dense()
model.to(device)

# count number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params} parameters')

#set a seed 
torch.manual_seed(0)

    
model.apply(init_weights)


optimizer = optim.Adam(model.parameters(), lr=0.001)

#take criterion and optimizer to device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion.to(device)



ckpt_root = f"{mlp_checkpoints_path}/dense/"
if not os.path.exists(ckpt_root): 
    os.makedirs(ckpt_root)

num_epochs = 200
train_mlp(model,num_epochs, ckpt_root, save=True)