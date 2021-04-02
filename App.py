# -*- coding: utf-8 -*-
"""

REQUIRED PYTHON PACKAGE INSTALLATION:
    PySimpleGUI: pip3 install pysimplegui
    PIL: 
    Numpy: 
    sklearn: 
    imgaug: pip3 install imgaug
    imageio: 
    pandas: 

Main application interface.

@author: Celina
"""
    
import PySimpleGUI as sg      # Simple GUI library
import numpy as np
import matplotlib as plt

from PIL import Image
from PIL.ImageOps import grayscale
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from preprocess import Data         

def applyPCA(im):
    """
        Processes an image with PCA and returns the reconstructed image.
        
        TODO: more proper implementation
        
    """
    # https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
    
    pca = PCA()
    pca.fit(im)
    
    # Getting the cumulative variance
    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
            
    # How many PCs explain 95% of the variance?
    k = np.argmax(var_cumu>95)
    #print("Number of components explaining 95% variance: "+ str(k))
    #print("\n")
            
    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(im))
    
    # Plotting the original image
    #plt.figure.Figure(figsize=[12,8])
    #plt.pyplot.imshow(im,cmap = plt.cm.gray)
    
    # Plotting the reconstructed image
    #plt.figure.Figure(figsize=[12,8])
    #plt.pyplot.imshow(image_recon,cmap = plt.cm.gray)
    
    return image_recon

def getBatches(lst, samples_per_batch):
    """Generator yielding successive chunks from lst, of size "samples_per_batch"."""
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        
    for i in range(0, len(lst), samples_per_batch):
        yield lst[i:i + samples_per_batch]

def loadBatchImages(batch, useGrayscale=False, pca=False, flatten=False):
    """
        Load a batch of image data, with optional processing.
        Returns a Numpy array of image matrices of shape (480, 640, 3) or (480, 640, 1) for grayscale,
        with optional flattening.
        
        batch - array of image filenames located in the 'augmented' directory
        useGrayscale - if True, converts the image to grayscale
        pca - if True, processes the image with PCA
        flatten - if True, flattens the image matrices
        
    """
    loadedBatch = []
    print("Loading image batch...")

    # Load images from this batch
    for im in batch:
        im = Image.open(augDir + "/" + im)
            
        # Additional processing
        if useGrayscale:
            im = grayscale(im)
        if pca:
            im = applyPCA(im)
        # Convert image to Numpy array
        im = np.asarray(im)
            
        # Flatten the image
        if flatten:
            im = im.flatten()
            
        # Store image and label 
        loadedBatch.append(im)

    return loadedBatch

def trainLoop(model, X_train, y_train, samples_per_batch, useGrayscale=False, pca=False, flatten=False):
    """

    Parameters
    ----------
    model : 
        Machine learning model to train
    X_train: Numpy array
        Contains filenames for the image data of the training set.
    y_train : Numpy array
        Contains labels for the training set data.
    samples_per_batch : int
        The number of samples to use per batch.
        
    useGrayscale : boolean, optional
        Converts image data to grayscale before testing. The default is False.
    pca : boolean, optional
        Processes images with PCA before testing. The default is False.
    flatten : boolean, optional
        Flattens the image data before feeding into the model. The default is False.

    Returns
    -------
    the trained model

    """
    
    # Split the training data and labels into batches
    batches = getBatches(X_train, samples_per_batch)
    batchLabels = getBatches(y_train, samples_per_batch)
    number_of_batches = len(list(getBatches(y_train, samples_per_batch)))
    
    i = 0
    # Train by iterating over batches
    for XBatch, yBatch in zip(batches, batchLabels):
        proceed = sg.OneLineProgressMeter("Train progress", i + 1, number_of_batches, "key", "Iterations indicate batches for training current model")
        
        if not proceed:
            print("Training was cancelled")
            model = None
            break
        else:
            XBatch = loadBatchImages(XBatch, useGrayscale=useGrayscale, pca=pca, flatten=flatten)
    
            # Partial fit on the current batch for training
            model.partial_fit(XBatch, yBatch, classes=[0,1,2,3,4])
            i += 1
            print("Batch {} complete for training current model".format(i))
        
    return model

def testLoop(model, X_test, y_test, samples_per_batch, useGrayscale=False, pca=False, flatten=False):
    """

    Parameters
    ----------
    model : 
        Machine learning model to score the test data on
    X_test : Numpy array
        Contains filenames for the image data of the test set.
    y_test : Numpy array
        Contains labels for the test set data.
    samples_per_batch : int
        The number of samples to use per batch.
        
    useGrayscale : boolean, optional
        Converts image data to grayscale before testing. The default is False.
    pca : boolean, optional
        Processes images with PCA before testing. The default is False.
    flatten : boolean, optional
        Flattens the image data before feeding into the model. The default is False.

    Returns
    -------
    The overall accuracy scores of this model (micro-averaged F1-score, general accuracy)

    """
    
    correct = 0      # Number of correct predictions
    totalTest = len(y_test)    # Total number of test samples
    
    # Split the test data (X_test) and labels (y_test) into batches
    batches = getBatches(X_test, samples_per_batch)
    batchLabels = getBatches(y_test, samples_per_batch)
    number_of_batches = len(list(getBatches(y_test, samples_per_batch)))
    
    i = 0

    # Test over all the batches
    for XBatch, yBatch in zip(batches, batchLabels):
        proceed = sg.OneLineProgressMeter("Test progress", i + 1, number_of_batches, "key", "Iterations indicate batches for testing current model")
        
        # Cancel if progress bar was cancelled
        if not proceed:
            print("Testing was cancelled")
            return None
        else:    
            XBatch = loadBatchImages(XBatch, useGrayscale=useGrayscale, pca=pca, flatten=flatten)
            
            # Score the model on this batch of test data
            accuracy = model.score(XBatch, yBatch)
            correct += accuracy*samples_per_batch
            
            ### TODO: F1-score
            print("Accuracy of batch: {}".format(accuracy))
            print("Number of correct predictions so far: {}".format(correct))
            i += 1
            print("Batch {} complete for testing current model".format(i))

    print("Overall accuracy: {}".format(correct/totalTest))

def trainBayes(X_train, X_test, y_train, y_test, random_state):
    print("Training Bayes model...")
    
    # Prepare Bayes model
    bayes = GaussianNB()
    
    # First do cross-validation on a batch(?) to estimate parameters
    # Try different var_smoothing values (default is 1e-9)
    # Test 10 candidates, using 5-fold cross validation by default
    # https://stackoverflow.com/questions/39828535/how-to-tune-gaussiannb
    #params_NB = {'var_smoothing': np.logspace(0,-9, num=2)}
    #gs_NB = GridSearchCV(estimator=bayes, 
    #                 param_grid=params_NB, 
    #                 verbose=1, 
    #                 scoring='f1_micro') 
    #gs_NB.fit(X_train, y_train)
    
    #params = gs_NB.best_params_
    #print(params)
    # Train using the best parameter
    # var_smoothing=params["var_smoothing"]
    
    # Train model in batches, applying grayscale, PCA and flattening the image data
    bayes = trainLoop(bayes, X_train, y_train, samples_per_batch, useGrayscale=True, pca=True, flatten=True)
    
    #### Cancel training quickly, TODO: make cancellation more proper
    if bayes is None:
        return
    
    print("Testing Bayes model...")
    testLoop(bayes, X_test, y_test, samples_per_batch, useGrayscale=True, pca=True, flatten=True)

def train(data):
    """
    Trains the Naive Bayes, SVM and CNN models, and evaluates them with test data.
    
    Returns:
        the best model according to micro-averaged F1 score
    
    """

    # Store final accuracies of each model across random states
    modelAccuracies = {
        "bayes": [],
        "svm": [],
        "cnn": []
        }
    
    # Train and run models with different random states
    ### limited to 1 value for now
    for random in range(0, 1):
        # Randomly split data into 80% training and 20% test
        X_train, X_test, y_train, y_test = data.splitData(random_state=random)
        
        # Train Gaussian Naive Bayes model
        bayesAcc = trainBayes(X_train, X_test, y_train, y_test, random)
        modelAccuracies["bayes"].append(bayesAcc)
        
        # Train SVM
        
        # Train CNN
        
    # Display and save results of training


def predict(image):
    """
    Uses the current model to predict what white blood cell is in the given image.

    Returns:
        The name of the white blood cell class, as a string

    """
    
    pass

def process_image(image):
    
    
    pass

def load_model(file):
    """
    Load an existing model from a file.
    
    """
    pass
    
augDir = r"augmented"

# Get original dataset
data = Data()

samples_per_batch = 500

#train(data)

# Set the layout of the GUI
MAX_INPUT_LENGTH = 5
layout = [[sg.Text("CPSC 599 - White Blood Cell Classifier")], 
          [sg.Text("Celina Ma, Rylan Marianchuk, Sam Robertson")],
          [sg.Button("Load Model"), sg.Button("Train Models"), sg.Text("Samples per Batch"), \
               sg.InputText(default_text=str(samples_per_batch), enable_events=True, size=(MAX_INPUT_LENGTH,1), key="-BATCHSIZE-")],
              
          [sg.Text("Current Model: None", key="--MODEL--")],
          [sg.In(size=(50, 1), enable_events=True, key="--IMAGE--"), \
           sg.FileBrowse(button_text="Select Image", file_types=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"),) )],
        
        [sg.Button("Predict Image", key="--PREDICT--")]
]

# Create the window
window = sg.Window("White Blood Cell Classifier", layout)

model = None

# Create an event loop
while True:
    # Obtain the latest event
    event, values = window.read()

    # End program if user closes window
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    
    elif event == "Load Model":
    # Select an existing model from a file
        window['--MODEL--'].update("Current Model: Test")
        
    elif event == '-BATCHSIZE-' and values['-BATCHSIZE-']:
        # Very basic input validation for batch size
        # Check input length
        if len(values['-BATCHSIZE-']) > 5:
            window.Element('-BATCHSIZE-').Update(values['-BATCHSIZE-'][:-1])
        else:
            # Check that field can be converted to int
            # https://pysimplegui.readthedocs.io/en/latest/cookbook/#recipe-input-validation
            try:
                in_as_int = int(values['-BATCHSIZE-'])
                samples_per_batch = in_as_int
            except:
                if len(values['-BATCHSIZE-']) == 1 and values['-BATCHSIZE-'][0] == '-':
                    continue
                window['-BATCHSIZE-'].update(values['-BATCHSIZE-'][:-1])
    
    elif event == "Train Models":
    # Retrain the models
        model = train(data)
        window['--MODEL--'].update("Current Model: Test")
        
    elif event == "--PREDICT--":
        # Run prediction on a user-provided image
        image = values["--IMAGE--"]
        
        # Load the image and process it
        if image != "":
            pass
        
            #if error:
            #    sg.Popup('Error message')
        elif image == "":
            sg.Popup('Please select an image to predict.')
        
window.close()
