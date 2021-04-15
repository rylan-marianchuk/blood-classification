# blood-classification

The main application can be run by executing 

python3 App.py

From the application window, the Bayes, SVM and CNN models can be trained.
After training a model, a results screen appears where models can be saved to a file.
The best model is saved to the main window as the Current Model, and can be used for
predictions.

The GoogLeNet model can be trained as a standalone executable, by executing:
python3 GoogLeNet.py

GoogLeNet was trained for comparison to the other models, but does not accept predictions.

Please note the CNN and GoogLeNet models have long training times and are hardware-intensive.
We include trained model files for Bayes, SVM and CNN for testing:
- bayes.pkl
- svm.pkl
- cnn.pkl

These can be loaded to the GUI with the "Load Model" button.

Github repository link: https://github.com/rylan-marianchuk/blood-classification