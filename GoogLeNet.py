"""
PyTorch implementation of Transfer learning with GoogLeNet

GoogLeNet weights are frozen, added a k neuron layer that is fully connected at end to learn

@author Rylan Marianchuk
"""


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np
from torchvision.models import googlenet
from preprocess import Data
import torchvision.transforms as transforms

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

# Classes
k = 4

# Pretrained GoogLeNet
model = googlenet(pretrained=True)
# Freezing weights
model.requires_grad_(False)

# Update the fully connected final layer to have k classes
model.fc = torch.nn.Linear(model.fc.in_features, k)

# Grab data and turn into dataset
data = Data(scale=0.55, normalize=True)
int_labels = torch.tensor(data.Y)
#all = CustomTensorDataset(tensors=(torch.tensor(data.X), int_labels), transform=
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
X = torch.tensor(data.X)
X_perm = X.permute(0, 3, 1, 2)
all = TensorDataset(X_perm, int_labels)
# Obs
n = len(all)
train_size = int(n*0.8)
test_size = n - int(n*0.8)
train, test = random_split(all, lengths=[train_size, test_size])

# Main train block
batch_size = 128
params = {
    'batch_size': batch_size,
    'shuffle': True
}
loader_tr = DataLoader(train, **params)
loader_te = DataLoader(test, **params)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_criteria = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.to("cuda")
    loss_criteria = loss_criteria.to("cuda")

# Train
model.train()
for epoch in range(30):
    # Store losses and accuracy on each batch
    loss_list = []
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader_tr):
        X, y = batch
        y = y.type(torch.LongTensor)
        X = X.to("cuda")
        y = y.to("cuda")
        optimizer.zero_grad()

        out = model(X)
        predicted = torch.max(out.data, 1)[1]
        loss = loss_criteria(out, y)
        loss_list.append(loss.item())
        correct += (predicted == y).sum()
        total += len(y)
        # Parameters update, the pytorch strength here
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: t-loss = {np.mean(loss_list):.4f}, t-acc = {correct/total:.4f}")

# Eval
model.eval()
running_correct = 0
total = 0
for batch_idx, batch in enumerate(loader_te):
    X, y = batch
    X = X.to("cuda")
    y = y.to("cuda")
    out = model(X)
    predicted = torch.max(out.data, 1)[1]
    running_correct += (predicted == y).sum()
    total += len(y)

print(f"Test Accuracy: {running_correct / total:.4f}")
