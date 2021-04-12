# -*- coding: utf-8 -*-
"""

REQUIRED PYTHON PACKAGE INSTALLATION:
    PySimpleGUI: pip3 install pysimplegui
    PIL: 
    Numpy: 
    sklearn: pip3 install sklearn
    imgaug: pip3 install imgaug
    imageio: 
    pandas: pip3 install pandas
    torch: pip3 install torch

Main application interface.

@author: Celina
"""
    
import PySimpleGUI as sg      # Simple GUI library
import numpy as np
import matplotlib.pyplot as plt
import pickle

from PIL import Image
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support

from preprocess import Data  

KEY_SCORE = "f1_macro"

def cross_validate(model, param_grid, X_valid, y_valid):
    """
    Perform 5-fold cross-validation on a model, using macro-averaged F1 score on a 
    provided validation set.

    Parameters
    ----------
    model : the sklearn model to validate on
    param_grid : dict
        Uses parameter names (str) for the model as keys, and maps them
        to a list of parameter settings to try.
        See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    X_valid : Numpy array
        The image data to validate over
    y_valid : Numpy array
        Labels for the image data
        
    Returns
    -------
    a dictionary mapping the parameter names to the settings
    which gave the best results on the validation set

    """
    
    # Perform cross validation on the given model, with 5-fold cross validation by default
    gridsearch = GridSearchCV(estimator=model, 
                     param_grid=param_grid, 
                     verbose=1, 
                     scoring=KEY_SCORE) 
    gridsearch.fit(X_valid, y_valid)
    
    print(gridsearch.best_params_)
    return gridsearch.best_params_

def train_test(model, X_train, y_train, X_test, y_test):
    """
        Train and test a provided model.
        
        Parameters
        ----------
        X_train : Numpy array
            Array of data to train on.
        X_test : Numpy array
            Labels for training data.
        y_train : Numpy array
            Array of data to test.
        y_test : Numpy array
            Labels for test data.
    
        Returns
        -------
        model :
            The trained model
        metrics : dict
            The final accuracy scores of the model (accuracy, precision, recall, f1)

    """
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    
    # Compute overall F1 score, precision, recall and accuracy
    precision, recall, f1score, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
    
    print("Overall accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Macro-averaged F1 score: {}".format(f1score))
    
    return (model, 
            {"accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "f1_macro": f1score})

def train_bayes(X_train, X_test, y_train, y_test):
    """
    Train the Gaussian Naive Bayes model and evaluate on the test set.

    Parameters
    ----------
    X_train : Numpy array
        Array of data to train on.
    X_test : Numpy array
        Labels for training data.
    y_train : Numpy array
        Array of data to test.
    y_test : Numpy array
        Labels for test data.

    Returns
    -------
    bayes :
        The trained Naive Bayes model
    metrics : dict
        The final accuracy scores of the model (accuracy, precision, recall, f1)
    params: dict
        The best parameters found for the model.

    """
    
    print("Training Bayes model...")
    
    # Prepare Bayes model
    bayes = GaussianNB()
    
    # Cross-validation to estimate var_smoothing value (default is 1e-9)
    # https://stackoverflow.com/questions/39828535/how-to-tune-gaussiannb
    param_grid = {'var_smoothing': np.logspace(0,-9, num=10)}

    params = cross_validate(bayes, param_grid, X_train, y_train)            
    print("Best parameters found: {}".format(params))

    # Train using the best parameters and evaluate
    bayes = GaussianNB(var_smoothing=params["var_smoothing"])
    bayes, metrics = train_test(bayes, X_train, y_train, X_test, y_test)
    
    return (bayes, metrics, params)

def process_record(model, model_data, metrics, params):
    """
    Processes a new model record.

    Parameters
    ----------
    model : 
        trained model object
    model_data : dictionary with keys "best_model",
        "best_params", "best_score", "scores"
    metrics : dict
        contains scores from recent model training
    params : dict
        contains parameters of the given model

    Returns
    -------
    None.

    """
    
    # Get the latest model score
    score = metrics[KEY_SCORE]
        
    # Update model data with training results
    model_data["scores"].append(score)
        
    # Update best model of this type
    if score > model_data["best_score"]:
        # Save new best model
        model_data["best_score"] = score
        model_data["best_model"] = model
        model_data["best_params"] = params

def train():
    """
    Trains the Naive Bayes, SVM and CNN models, and evaluates them with test data.
    
    Returns:
        the best model and its name
    
    """

    # Store record of each model across random states
    model_record = {
        "Bayes": {"best_model": None,   # Best Bayes model + parameters
                  "best_params": None,
                  "best_score": 0,
                  "scores": []},    # List of scores for each run
        "SVM": {"best_model": None,
                  "best_params": None,
                  "best_score": 0,
                  "scores": []},
        "CNN": {"best_model": None,
                  "best_params": None,
                  "best_score": 0,
                  "scores": []}
        }
    
    data = Data(scale=0.25)
    
    # Train and run models with different random states
    ### limited to 1 value for now
    for random in range(0, 1):
        # Randomly split data into 80% training and 20% test
        X_train, X_test, y_train, y_test = data.splitData(random_state=random)
        #print(len(X_train))
        #print(len(X_test))
        
        # Train CNN with colour images
        
        # Train Bayes and SVM with grayscale and flattening
        X_train = data.extra_processing(X_train, grayscale=True, flatten=True)
        X_test = data.extra_processing(X_test, grayscale=True, flatten=True)
        
        # Train Bayes and record metrics
        bayes, metrics, params = train_bayes(X_train, X_test, y_train, y_test)
        process_record(bayes, model_record["Bayes"], metrics, params)
    
    # TODO: GUI
    # Display results of training
    print("Overall results: {}".format(model_record))
    
    # Determine the best model by comparing average scores
    best_average = 0
    best_model_name = None
    best_params = None
    best_model = None
    
    # Loop over each type of model
    for model_name, record in model_record.items():
        if record["scores"]:
            av = sum(record["scores"])/len(record["scores"])   # Model's average score
            if av > best_average:
                best_average = av
                best_model_name = model_name
                best_params = record["best_params"]
                best_model = record["best_model"]
    
    # Show best model name and parameters
    print("Best model: {}".format(best_model_name))
    print("Average accuracy: {}".format(best_average))
    print("Parameters: {}".format(best_params))
    
    return (best_model, best_model_name)

def predict(image_file, model, data):
    """
    Uses the current model to predict what white blood cell is in the given image.

    Returns:
        The name of the white blood cell class, as a string

    """
    
    # Load image
    im = Image.open(r"augmented/" + image_file)
    width, height = im.size
    
    # Check that it matches the required dimensions
    if width != 640 or height != 480:
        sg.Popup("Selected image does not have 640x480 dimensions")
    else:
        # Process the image as needed, according to the model
        #im = data.transform_data([im])
        
        # Run model and show prediction
        predictions = model.predict([im])
        class_name = Data.class_map[predictions[0]]
        sg.Popup("The predicted class is {}".format(class_name))

def load_model(file):
    """
    Load an existing model from a file.
    
    Parameters
    file : string
        The path of the file where the model is saved.
    
    """
    model = pickle.load(open(file))
    
    return model
    
def save_model(model):
    """
    
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pass



# Get original dataset
#data = Data()

#bayes = GaussianNB()
#pickle.dump(bayes, open(filename, "wb" ))

# Set the layout of the GUI
layout = [[sg.Text("CPSC 599 - White Blood Cell Classifier")], 
          [sg.Text("Celina Ma, Rylan Marianchuk, Sam Robertson")],
          [sg.FileBrowse(button_text="Load Model", key="--LOAD--"), \
           sg.Button("Train Models")],
              
          [sg.Text("Current Model: None", key="--MODEL--")],
          [sg.In(size=(50, 1), enable_events=True, key="--IMAGE--"), \
           sg.FileBrowse(button_text="Select Image", file_types=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"),) )],
        
        [sg.Button("Predict Image", key="--PREDICT--")],
        
        [sg.Button("Test")]
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
        file = window["--LOAD--"]
        model, model_name = load_model(file)
        window['--MODEL--'].update("Current Model: {}".format(model_name))
    
    elif event == "Train Models":
    # Retrain the models
        model, model_name = train()
        window['--MODEL--'].update("Current Model: {}".format(model_name))
        
    elif event == "--PREDICT--":
        # Run prediction on a user-provided image
        image = values["--IMAGE--"]
        
        # Load the image and process it
        if image != "":
            predict(image, model)
        
            #if error:
            #    sg.Popup('Error message')
        elif image == "":
            sg.Popup('Please select an image to predict.')
            
    elif event == "Test":
        layout2 = [
        [sg.FileSaveAs(button_text="Save Bayes Model", key="-SAVEBAYES-")],
        [sg.FileSaveAs(button_text="Save SVM Model", key="-SAVESVM-")],
        [sg.FileSaveAs(button_text="Save CNN Model", key="-SAVECNN-")]
        ]

        newWindow = sg.Window("Model Results", layout2)
        
        event, values = newWindow.Read()

        if event == '-SAVEBAYES-':
            file = values["-SAVEBAYES"]
            if file:
                save_model("bayes")
        
window.close()
