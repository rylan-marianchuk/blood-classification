# -*- coding: utf-8 -*-
"""

REQUIRED PYTHON PACKAGE INSTALLATION:
    PySimpleGUI: pip3 install pysimplegui
    PIL: 
    Numpy: 
    sklearn: 

Main application interface.

@author: Celina
"""
    
import PySimpleGUI as sg      # Simple GUI library
import os
import numpy as np
import matplotlib as plt

from PIL import Image
from PIL.ImageOps import grayscale
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from preprocess import Data

DATAPATH = "D:/dataset/599augmented/"


"""def compressData():
    COMPRESSEDPATH = "D:/dataset/599compressed/"
    
    images = os.listdir(DATAPATH)
    totalImages = len(images)
        
    for filename in images:
        im = Image.open(DATAPATH + filename)
        #sg.OneLineProgressMeter('Progress', len(data) + 1, totalImages, 'key')
            
        if compress:
            im = im.resize((int(640/4),int(480/4)), Image.ANTIALIAS, optimize=True)"""


LABEL_NAMES = ["NEUTROPHIL", "BASOPHIL", "EOSINOPHIL", "MONOCYTE", "LYMPHOCYTE"]

def getData(dataFolder, useGrayscale=True, pca=False):
    
    """
        Loads images from the dataset for training/testing.
        Optionally processes images into grayscale and applies PCA
        dimension reduction.
        
        Returns:
            data: a Numpy array of features for the image data
            labels: a list of blood cell labels for each image in the dataset
    
    """
    
    data = []
    labels = []

    # Load all images into Numpy arrays and load their labels
    # Mapping:
        # 0 - Neutrophil
        # 1 - Basophil
        # 2 - Eosinophil
        # 3 - Monocyte
        # 4 - Lymphocyte
    images = os.listdir(dataFolder)
    totalImages = len(images)
        
    for filename in images:
        im = Image.open(dataFolder + filename)
        #sg.OneLineProgressMeter('Progress', len(data) + 1, totalImages, 'key')
        
        # Process image into grayscale
        if useGrayscale:
            im = grayscale(im)            
            
        # Reduce image dimensions with PCA
        # https://www.kaggle.com/mirzarahim/introduction-to-pca-image-compression-example
        if pca:
            pca = PCA(n_components=18)
            pca.fit(im)

            # Getting the cumulative variance
            var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
            
            # How many PCs explain 95% of the variance?
            k = np.argmax(var_cumu>95)
            #print("Number of components explaining 95% variance: "+ str(k))
            #print("\n")
            
            ipca = IncrementalPCA(n_components=k)
            image_recon = ipca.inverse_transform(ipca.fit_transform(im))
            
            # Plotting the reconstructed image
            #plt.figure.Figure(figsize=[12,8])
            #plt.pyplot.imshow(im,cmap = plt.cm.gray)
        
        # Store image data
        data.append(np.asarray(im).flatten())
        
        # Store label of the image
        for index, label in enumerate(LABEL_NAMES):
            if label in filename:
                labels.append(index)
        
        ### Cut dataset short temporarily
        if len(data) + 1 == 500:
            break
        
    # Convert data to NP array and return
    data = np.array(data)
    
    return data, labels

def trainBayes(X_train, X_test, y_train, y_test, random_state):
    # Prepare Bayes model
    bayes = GaussianNB(verbose=1)
    
    # Try different var_smoothing values (default is 1e-9)
    # Test 10 candidates, using 5-fold cross validation by default
    # https://stackoverflow.com/questions/39828535/how-to-tune-gaussiannb
    params_NB = {'var_smoothing': np.logspace(0,-9, num=2)}
    gs_NB = GridSearchCV(estimator=bayes, 
                     param_grid=params_NB, 
                     verbose=1, 
                     scoring='f1_micro') 
    gs_NB.fit(X_train, y_train)
    
    params = gs_NB.best_params_
    print(params)
    
    # Train using the best parameter
    bayes = GaussianNB(var_smoothing=params["var_smoothing"]).fit(X_train, y_train)
    accuracy = bayes.score(X_test, y_test)
    f1score = bayes.score(X_test, y_test)
        
    # Make prediction on test set
    print("Bayes accuracy on test set with random_state={} and var_smoothing={}: {:.3f}".format(random_state, \
                                                                                        params["var_smoothing"], accuracy))

def train(data, labels):
    """
    Runs the Naive Bayes, SVM and CNN models on the training dataset.
    
    Returns:
        the best model according to micro-averaged F1 score
    
    """
    
    # Preprocess dataset
    
    # Store final accuracies of each model across random states
    modelAccuracies = {
        "bayes": [],
        "svm": [],
        "cnn": []
        }
    
    # Consider including a progress bar
    
    # Train and run models with different random states
    for random in range(0, 10):
        # Randomly split data into 80% training and 20% test
        X_train, X_test, y_train, y_test = train_test_split(
        data, labels, stratify=labels, test_size=0.2)
        
        # Train Gaussian Naive Bayes model
        bayesAcc = trainBayes(X_train, X_test, y_train, y_test, random)
        modelAccuracies["bayes"].append(bayesAcc)
        
        # Train SVM
        
        # Train CNN
        
        break
    
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


data, labels = getData(DATAPATH, useGrayscale=True)

# Set the layout of the GUI
layout = [[sg.Text("CPSC 599 - White Blood Cell Classifier")], 
          [sg.Text("Celina Ma, Rylan Marianchuk, Sam Robertson")],
          [sg.Button("Load Model"), sg.Button("Train Models")],
          [sg.Text("Current Model: None", key="--MODEL--")],
          
          [sg.In(size=(50, 1), enable_events=True, key="--IMAGE--"), \
           sg.FileBrowse(button_text="Select Image", file_types=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"),) )],
        
        [sg.Button("Predict Image", key="--PREDICT--")]
]

# Create the window
window = sg.Window("White Blood Cell Classifier", layout)

model = None

# Train models
train(data, labels)

# Create an event loop
while True:
    # Obtain the latest event
    event, values = window.read()
    #print(event)
    
    # End program if user closes window
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    elif event == "Load Model":
    # Select an existing model from a file
        window['--MODEL--'].update("Current Model: Test")
    
    elif event == "Train Models":
    # Retrain the models
        train()
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
        
        # Train on the current model

window.close()
