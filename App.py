# -*- coding: utf-8 -*-
"""

REQUIRED PYTHON PACKAGE INSTALLATION:
    PySimpleGUI: pip3 install pysimplegui
    PIL: pip3 install Pillow
    Numpy: pip3 install numpy
    sklearn: pip3 install sklearn
    imgaug: pip3 install imgaug
    imageio: pip3 install imageio
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
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, f1_score

from unet import train_unet, finetune_cnn
import torch

from preprocess import Data


KEY_SCORE = "f1_macro"
BAYES_SCALE = 0.25

# For GUI fields
MAX_INPUT_LENGTH = 3
DEFAULT_LOOPS = 1

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
    
    # Train and score the model
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


def train_svm(X_train, X_test, y_train, y_test, seed):
    """
    Train the SVM and evaluate on the test set.
    """
    print('Training SVM...')
    svm = SVC(verbose=0, max_iter=-1)
    # sigmoid and linear kernels achieve poorer results
    gridsearch = GridSearchCV(estimator=svm,
                              param_grid={'kernel': ['poly', 'rbf'],
                                          'C': [10, 100],
                                          'tol': [1e-3],
                                          'random_state': [seed]},
                              verbose=0,
                              scoring='f1_macro')
    # select the best model via cross validation
    best_model = gridsearch.fit(X_train, y_train)
    # save the hyperparameters selected for the best model
    params = gridsearch.best_params_
    # compute the scores for the best model
    acc = best_model.score(X_test, y_test)
    pred = best_model.predict(X_test)
    f1_macro = f1_score(y_test, pred, average="macro")
    metrics = {'accuracy': acc, 'f1_macro': f1_macro}
    print('Done Training SVM')
    return best_model, metrics, params

def process_record(model, model_data, metrics, params):
    """
    Processes a new model record and updates the model's data.

    Parameters
    ----------
    model : 
        trained model object
    model_data : dictionary with keys "best_model",
        "best_params", "best_score", "main_scores"
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
    model_data["metrics"].append(metrics)
        
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
                  "metrics": []},    # List of metrics for each run
        "SVM": {"best_model": None,
                  "best_params": None,
                  "best_score": 0,
                  "metrics": []},
        "CNN": {"best_model": None,
                  "best_params": None,
                  "best_score": 0,
                  "metrics": []}
        }
    
    print("Loading and augmenting data...")
    data = Data(scale=BAYES_SCALE, dups=25)
    
    # Train and run models with different random states
    for random in range(0, num_loops):
        # Randomly split data into 80% training and 20% test
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = data.splitData(random_state=random)

        # Train CNN with colour images
        cnn, metrics, params = finetune_cnn(epochs=1, seed=random)
        process_record(cnn, model_record["CNN"], metrics, params)

        # Train Bayes and SVM with grayscale and flattening
        X_train = Data.extra_processing(X_train, grayscale=True, flatten=True)
        X_test = Data.extra_processing(X_test, grayscale=True, flatten=True)

        # Train Bayes and record metrics
        bayes, metrics, params = train_bayes(X_train, X_test, y_train, y_test)
        process_record(bayes, model_record["Bayes"], metrics, params)

        # Train SVM and record metrics
        # Reduce the training and test set, keeping the 80/20 split
        # The svm doesn't improve significantly with more augmented data,
        # and it's training time si O(n^2) in the number of training samples
        svm_X_train = X_train[:500]
        svm_y_train = y_train[:500]
        svm_X_test = X_test[:100]
        svm_y_test = y_test[:100]
        svm, metrics, params = train_svm(svm_X_train, svm_X_test, svm_y_train, svm_y_test, random)
        process_record(svm, model_record["SVM"], metrics, params)

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
        if record["metrics"]:
            # Compute model's average F1 score
            total = 0
            for entry in record["metrics"]:
                total += entry[KEY_SCORE]
            
            av = total/len(record["metrics"])   
            if av > best_average:
                best_average = av
                best_model_name = model_name
                best_params = record["best_params"]
                best_model = record["best_model"]
    
    # Show best model name and parameters
    print("Best model: {}".format(best_model_name))
    print("Average macro F1-score: {}".format(best_average))
    print("Parameters: {}".format(best_params))
    
    # Let user save models to file
    save_model_popup(model_record)
    
    return (best_model, best_model_name)

def predict(image_file, model):
    """
    Uses the current model to predict what white blood cell is in the given image.

    Returns:
        The name of the white blood cell class, as a string

    """
    
    # Load image
    im = Image.open(image_file)
    width, height = im.size
    
    # Check that it matches the required dimensions
    if width != 640 or height != 480:
        sg.Popup("Selected image does not have 640x480 dimensions")
    else:
        # Process the image as needed, according to the model
        model_class = model.__class__.__name__
        image_list = []
        print(model_class)
        
        print("Processing image sample...")
        
        if model_class == "GaussianNB":
            # Scale down, convert image to grayscale and flatten
            scale = BAYES_SCALE
            im = im.resize((int(im.size[0]*scale), int(im.size[1]*scale)), Image.BICUBIC)
            im = np.array(im)
            image_list.append(im)
            image_list = np.array(image_list)
            image_list = Data.extra_processing(image_list, grayscale=True, flatten=True)

        # Run model and show prediction
        predictions = model.predict(image_list[0].reshape(-1, 1))
        class_name = Data.class_map[predictions[0]]
        sg.Popup("The predicted class is {}".format(class_name))
        

def load_model(file):
    """
    Load an existing model from a file.
    
    Parameters
    file : string
        The file where the model is saved.
        
    Returns:
        the model object and the model name
    
    """
    
    model = None
    model_name = None

    if file:
        with open(file, "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
            model_name = model_data["model_name"]

    return (model, model_name)


def save_model_popup(model_record):
    """
    Creates a popup window for saving the trained models
    and displaying training results.

    Parameters
    ----------
    model_record : dict
        Stores model information.

    Returns
    -------
    None.

    """
    
    # TODO: show the training results in GUI
    # Make layout for save popup
    layout2 = [        
        [sg.Input(visible=False, enable_events=True, key="-SAVEBAYES-"), sg.FileSaveAs(button_text="Save Bayes Model", target="-SAVEBAYES-")],
        [sg.Input(visible=False, enable_events=True, key="-SAVESVM-"), sg.FileSaveAs(button_text="Save SVM Model", target="-SAVESVM-")],
        [sg.Input(visible=False, enable_events=True, key="-SAVECNN-"), sg.FileSaveAs(button_text="Save CNN Model", target="-SAVECNN-"),]
        ]

    newWindow = sg.Window("Model Results", layout2)
    
    while True:
        event, values = newWindow.Read()
        
        # User chooses which model to save
        # End program if user closes window
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        elif event == '-SAVEBAYES-':
            file = values["-SAVEBAYES-"]
            save_model("Bayes", model_record["Bayes"]["best_model"], file)
            
        elif event == "-SAVESVM-":
            file = values["-SAVESVM-"]
            save_model("SVM", model_record["SVM"]["best_model"], file)
        elif event == "-SAVECNN-":
            file = values["-SAVECNN-"]
            save_model("CNN", model_record["CNN"]["best_model"], file)


def save_model(model_name, model, file):
    """
    Save the model name and object to a file
    in dictionary format.

    Parameters
    ----------
    model_name : String name of model (eg. Bayes)

    model : the model object

    file: String name of file to save to

    Returns
    -------
    None.

    """
    if file:
        # if model_name == "CNN":
        #     torch.save(model.state_dict(), file)
        # else:
        # Save model name and object as dictionary
        model_data = {"model_name": model_name,
                      "model": model}
        pickle.dump(model_data, open(file, "wb"))
        sg.Popup("Model saved!")



# Set the layout of the GUI
layout = [[sg.Text("CPSC 599 - White Blood Cell Classifier")], 
          [sg.Text("Celina Ma, Rylan Marianchuk, Sam Robertson")],
          [sg.Input(key='-LOADEDFILE-', visible=False, enable_events=True), \
           sg.FileBrowse(button_text="Load Model", target="-LOADEDFILE-", \
                         file_types=(("All files", "*.*"), ("No extension", ""), ("ALL files", "*")), key="--LOAD--"), \
           sg.Button("Train Models"),
           sg.Text("Number of Random States"), sg.Input(default_text=str(DEFAULT_LOOPS), enable_events=True, \
                                                        size=(MAX_INPUT_LENGTH,1),  key='-LOOPS-')],
          [sg.Text("Current Model: None", key="--MODEL--")],
          [sg.In(size=(50, 1), enable_events=True, key="--IMAGE--"), \
           sg.FileBrowse(button_text="Select Image", file_types=(("PNG Files", "*.png"), ("JPG Files", "*.jpg"),) )],
        
        [sg.Button("Predict Image", key="--PREDICT--")]
]

# Create the window
window = sg.Window("White Blood Cell Classifier", layout)
model = None
model_name = None
num_loops = DEFAULT_LOOPS

# Create an event loop
while True:
    # Obtain the latest event
    event, values = window.read()
    

    # End program if user closes window
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    
    elif event == "-LOADEDFILE-":
    # Select an existing model from a file
        file = values["-LOADEDFILE-"]
        model, model_name = load_model(file)
        window['--MODEL--'].update("Current Model: {}".format(model_name))
    
    elif event == "Train Models":
    # Retrain the models
        model, model_name = train()
        window['--MODEL--'].update("Current Model: {}".format(model_name))
        
    elif event == '-LOOPS-' and values['-LOOPS-']:
        # Very basic input validation for number of loops
        # Check input length
        if len(values['-LOOPS-']) > 3:
            window.Element('-LOOPS-').Update(values['-LOOPS-'][:-1])
        else:
            # Check that field can be converted to int
            # https://pysimplegui.readthedocs.io/en/latest/cookbook/#recipe-input-validation
            try:
                in_as_int = int(values['-LOOPS-'])
                num_loops = in_as_int
            except:
                if len(values['-LOOPS-']) == 1 and values['-LOOPS-'][0] == '-':
                    continue
                window['-LOOPS-'].update(values['-LOOPS-'][:-1])
        
    elif event == "--PREDICT--":
        # Run prediction on a user-provided image
        image = values["--IMAGE--"]
        
        # Load the image and process it
        if image != "":
            predict(image, model)

        elif image == "":
            sg.Popup('Please select an image to predict.')
        
window.close()
