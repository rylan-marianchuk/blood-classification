# -*- coding: utf-8 -*-
"""

REQUIRED PYTHON PACKAGE INSTALLATION:
    PySimpleGUI: pip3 install pysimplegui

Main application interface.

@author: Celina
"""
    
import PySimpleGUI as sg      # Simple GUI library

def train():
    """
    Runs the Naive Bayes, SVM and CNN models on the training dataset.
    
    Returns:
        the best model according to micro-averaged F1 score
    
    """
    
    # Preprocess dataset
    
    # Train and run models, storing evaluation data
    
    # Display and save results of training
    
    pass

    # Consider including a progress bar

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
train()

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
