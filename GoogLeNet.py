"""
PyTorch implementation of Transfer learning with GoogLeNet

GoogLeNet weights are frozen, added a k neuron layer that is fully connected at end to learn

@author Rylan Marianchuk
"""


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from torchvision.models import googlenet
from preprocess import Data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
# Load the datasets
loader_tr = DataLoader(train, **params)
loader_te = DataLoader(test, **params)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_criteria = torch.nn.CrossEntropyLoss()

# Attach to gpu
if torch.cuda.is_available():
    model = model.to("cuda")
    loss_criteria = loss_criteria.to("cuda")

# Train
model.train()
for epoch in range(12):
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

        # Get loss
        loss = loss_criteria(out, y)
        loss_list.append(loss.item())
        # Update accuracy
        correct += (predicted == y).sum()
        total += len(y)
        # Parameters update after calculating loss, pytorch strength here
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: t-loss = {np.mean(loss_list):.4f}, t-acc = {correct/total:.4f}")

# Eval
model.eval()
running_correct = 0
total = 0
predicted_all = []
true_all = []
# Analogous to train loop, only loading test set
for batch_idx, batch in enumerate(loader_te):
    X, y = batch
    X = X.to("cuda")
    y = y.to("cuda")
    out = model(X)
    predicted = torch.max(out.data, 1)[1]
    running_correct += (predicted == y).sum()
    total += len(y)
    predicted_all += predicted.tolist()
    true_all += y.tolist()

print(f"Test Accuracy: {running_correct / total:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

# Obtaining confusion matrix and f1 score metrics

f1_macro = f1_score(true_all, predicted_all, average='macro')
print(f1_macro)
cm = confusion_matrix(true_all, predicted_all, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["EOSIN", "LYMPH", "MONO", "NEUTRO"])
disp.plot()
plt.show()
