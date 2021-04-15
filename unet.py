from preprocess import Data
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
import numpy as np
import random
'''
Pytorch coding style inspired by/adapted from:
 https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

@author Sam
'''


# As in https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
def double_conv(in_channels, out_channels):
    '''
    A frequently used layer consisting of two convolutional convolutional layers,
    each followed by a ReLU activation function. This makes up the basic building
    block of the Unet architecture.
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )


class Unet(nn.Module):
    '''
    Architecure based on unet, popular for semantic segmentation in biomedical
    applications. Adapted to be smaller and with added fully connected layers
    at the end to convert to one-hot encoded probablistic class scores.
    Uses dropout.
    Adapted from the architecture of
    https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
    '''

    def __init__(self):
        # Sets up all the memory needed for the models trainable parameters
        super(Unet, self).__init__()
        # conv down layers
        self.down1 = double_conv(3, 32)
        self.down2 = double_conv(32, 64)
        self.down3 = double_conv(64, 128)
        # pooling and upsampling
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # conv up layers
        self.up1 = double_conv(64 + 128, 64)
        self.up2 = double_conv(32 + 64, 32)
        self.last_conv = nn.Conv2d(32, 4, 1)
        # fc layers
        self.fully_connected = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(4*264*264, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        # computes a forward pass through the neural network, given input data x
        # block 1
        conv1 = self.down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)
        # block 2
        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        x = self.dropout(x)
        # middle
        x = self.down3(x)
        x = self.dropout(x)
        # block 3
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dropout(x)
        # block 4
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dropout(x)
        # block 5
        x = self.up2(x)
        x = self.dropout(x)
        # convolve to class probabilities
        x = self.last_conv(x)
        # output
        out = self.fully_connected(x)
        return out


def train(dataloader, model, loss_fn, optim, device):
    '''
    Train a CNN for a single epoch.

    Parameters
    ---------

    dataloader : the dataset to train on, wrapped as a generator of pytorhc tensors.

    model : the instantiated model to train

    loss_fn : a pytorch loss function

    optim : a pytorch optimizer

    device : whether or not to use a cuda-enabled gpu
    '''
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # forward pass
        pred = torch.squeeze(model(X))
        # loss computation
        loss = loss_fn(pred, y)
        # backprop
        optim.zero_grad()
        loss.backward()
        optim.step()
        # print output
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print('loss: {} [{}/{}]'.format(loss, current, size))


# this development function prints and returns test set metrics
def test(dataloader, model, loss_fn, device):
    '''
    Evaluate a trained model on a testing dataset.

    Parameters
    ---------

    dataloader : the dataset to train on, wrapped as a generator of pytorhc tensors.

    model : the trained model to evaluate

    loss_fn : a pytorch loss function

    device : whether or not to use a cuda-enabled gpu

    Returns
    -------

    correct : the accuracy of the model

    loss : the loss of the model

    f1_macro : the macroaveraged F1 score of the model
    '''
    size = len(dataloader.dataset)
    model.eval()
    loss, correct = 0, 0
    pred_list = []
    y_true = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)    # Model prediction on the loaded data
            loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            correct += (pred == y).type(torch.float).sum().item()  # Track model score
            # accumulate info for f1 score
            pred_list.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    loss /= size
    correct /= size
    f1_macro = f1_score(y_true, pred_list, average='macro')
    print('Test Metrics:\n Accuracy: {}%\n Avg loss: {}\n Macroaveraged F1: {}\n'.format(100*correct, loss, f1_macro))
    return correct, loss, f1_macro


# load the data as required for CNN training
def load_data(val_proportion=0.1, scale=0.55, batch_size=16):
    '''
    Create data loaders for the training and test set.

    Parameters
    ----------

    val_proportion : the amount of the trianing data to partition off as a validation set

    scale : the proportion of the original image to resize to

    batch_size : the number of images to generate in a batch

    Returns
    -------

    train_dl : a dataloader object for the training set

    test_dl : a dataloader object for the test set
    '''
    print('Loading Dataset for CNN')
    data = Data(scale=scale, normalize=True)  # Load rescaled 55% images
    X = torch.tensor(data.X).permute(0, 3, 1, 2)  # Convert to tensors
    y = torch.tensor(data.Y)
    data = TensorDataset(X, y)
    n = len(data)
    train_size = int(n * val_proportion)
    test_size = n - train_size
    train_data, test_data = random_split(data, lengths=[train_size, test_size])
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


# This method performs hyperparameter search
def finetune_cnn(verbose=False, epochs=2, seed=0):
    '''
    Train the model for multiple emacs, doing a grid search over hyperparameters.

    Parameters
    ----------

    verbose : whether or not to give a detailed printout of the finetuning process

    epochs : the number of epochs to train each model for

    seed : the random seed to use for training

    Returns
    -------

    best_model : the most performant trained model

    metrics : the scores of the best model

    best_model_params : the hyperparameters which generated the best model
    '''
    print('Training CNN...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print('Using {} device'.format(device))
    best_model = None
    best_model_score = -1
    best_model_params = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_dl, test_dl = load_data(val_proportion=0.2, scale=0.1, batch_size=16)
    # These class weights attempt to prevent the network from learning to
    # only predict the majority class
    class_weights = torch.Tensor([1, 1, 1, 0.6])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    # Validate over different learning rates and optimizers
    for learning_rate in [1e-4, 1e-3]:
        for optimizer_name in ['adam', 'sgd']:
            model = Unet().to(device)
            if optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.train()
            if verbose:
                print('Model Training Beginning')
            for t in range(epochs):
                if verbose:
                    print(f'Epoch {t+1}\n--------------------------------')
                train(train_dl, model, loss_fn, optimizer, device)
                acc, loss, f1 = test(test_dl, model, loss_fn, device)
            if verbose:
                print('Model Training Finished')
            # Store best model
            if f1 > best_model_score:
                model.cpu()
                best_model = model
                best_model_params = {'optimizer': optimizer_name,
                                     'learning_rate': learning_rate}
    best_model.to(device)
    # Run model on test set and return final scores
    metrics = test_unet(test_dl, best_model, loss_fn, device)
    best_model.cpu()
    return best_model, metrics, best_model_params


# function to evaluate a unet on a given dataset, returns dictionary
def test_unet(dataloader, model, loss_fn, device):
    '''
    Evaluate a trained model on a testing dataset.

    Parameters
    ---------

    dataloader : the dataset to train on, wrapped as a generator of pytorhc tensors.

    model : the trained model to evaluate

    loss_fn : a pytorch loss function

    device : whether or not to use a cuda-enabled gpu

    Returns
    -------

    A dictionary containing the best model accuracy and f1 score.
    '''
    size = len(dataloader.dataset)
    model.eval()
    loss, correct = 0, 0
    pred_list = []
    y_true = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)   # Make predictions and score
            loss += loss_fn(pred, y).item()
            pred = torch.argmax(pred, dim=1)
            correct += (pred == y).type(torch.float).sum().item()
            # accumulate info for f1 score
            pred_list.extend(pred.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    
    # Return model's F1-score and accuracy
    acc = correct / size
    f1_macro = f1_score(y_true, pred_list, average='macro')
    return {'accuracy': acc, 'f1_macro': f1_macro}


# this function can be used to train a single (the best) unet only
def train_unet():
    '''
    Trains the model with the best parameters found (development function).

    Returns
    -------

    model : the trained model

    metrics : the scores of the model

    params : the hyperparameters which generated the model
    '''
    print('Training CNN...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    train_dl, test_dl = load_data(val_proportion=0.2, scale=0.1, batch_size=16)
    model = Unet().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 5
    model.train()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dl, model, loss_fn, optimizer, device)
    metrics = test_unet(test_dl, model, loss_fn, device)
    params = {'optimizer': 'adam', 'loss': 'crossentropy', 'lr': 1e-4}
    print('Done Training CNN')
    return model, metrics, params
