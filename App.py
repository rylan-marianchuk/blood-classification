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

def train():
    """
    Trains the Naive Bayes, SVM and CNN models, and evaluates them with test data.
    
    Returns:
        the best model according to micro-averaged F1 score
    
    """

    # Store final accuracies of each model across random states
    model_record = {
        "bayes": [],
        "svm": [],
        "cnn": []
        }
    
    data = Data(scale=0.25)
    
    # Train and run models with different random states
    ### limited to 1 value for now
    for random in range(0, 1):
        # Randomly split data into 80% training and 20% test
        X_train, X_test, y_train, y_test = data.splitData(random_state=random)
        
        # Train CNN with colour images
        
        # Train Bayes and SVM with grayscale and flattening
        X_train = data.extra_processing(X_train, grayscale=True, flatten=True)
        X_test = data.extra_processing(X_test, grayscale=True, flatten=True)
        
        #print(len(X_train))
        #print(len(X_test))
        
        bayes, bayes_metrics, params = train_bayes(X_train, X_test, y_train, y_test, random)
        model_record["bayes"].append({"model": bayes,
                                         "metrics": bayes_metrics,
                                         "params": params})
    
    # Display results of training
    print(model_record)
    
    # Determine the best model
    averages = []         # Average accuracies for each model type across runs
    best_models = []      # Best model of each type
    
    for m in model_record:
        total_score = 0
        i = 0
        best_model = None
        best_params = None
        best_score = 0     # Track the highest score for this model type
        
        for record in m:
            # Sum the score from each run
            score = record[i]["metrics"][KEY_SCORE]
            total_score += score
            i += 1
            
            # Track the best-performing model of each type
            if best_model is None or score > best_score:
                best_model = record[i]["model"]
                best_params = record[i]["params"]
        
        # Average score for this model type
        average = total_score/i
        averages.append(average)
        best_models.append(best_model)
    
    # Get index of model with best average accuracy
    model_index = averages.index(max(averages))
    
    return best_models[model_index]

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

#filename = "blah"

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
        file = window["--"]
        window['--MODEL--'].update("Current Model: Test")
    
    elif event == "Train Models":
    # Retrain the models
        model = train()
        window['--MODEL--'].update("Current Model: Test")
        
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
