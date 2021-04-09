import pandas as pd
import imgaug as ia
import os
import imageio
import numpy as np
from PIL import Image

cream_bg = (195, 201, 198)

# Euclidean distance between two vectors
def dist(vec1, vec2):
    return np.sqrt(sum((vec1[i]-vec2[i])**2 for i in range(len(vec1))))

def mask(img):
    # If all channels are below some threshold, turn to cream
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Mask all but blue
            if dist((0, 0, 0), img[i][j]) < 45:
              img[i][j][0] = cream_bg[0]
              img[i][j][1] = cream_bg[1]
              img[i][j][2] = cream_bg[2]

    #ia.imshow(img)


def augment():
    """
    Need only be run once

    Augmentations to apply:

    :return: Create a new directory with augmented imaged using imgaug
    """

    # Reading csv for label
    df = pd.read_csv(r"dataset-master/dataset-master/labels.csv")
    label_col = df["Category"].tolist()
    labels = []
    inDIR = r"dataset-master/dataset-master/JPEGImages"
    count = 0
    images = []
    for im in os.listdir(inDIR):
        if type(label_col[count]) == type(float()) or len(label_col[count].split(" ")) != 1:
            # 0 or more than one label, discard
            count += 1
            continue

        imagePIL = Image.open(inDIR + "/" + im)
        image_as_array = np.array(imagePIL)
        for _ in range(28):
            images.append(image_as_array)
            labels.append(label_col[count])
        count += 1
    print("COUNT (number of original images duplicated 28 times: " + str(count))
    print("TOTAL images to create: " + str(count * 28))

    flipH = ia.augmenters.Fliplr(0.3333)
    flipV = ia.augmenters.Flipud(0.3333)
    one = ia.augmenters.Sometimes(0.25, ia.augmenters.Affine(
            scale={"x": (0.8, 1.12), "y": (0.8, 1.12)}))
    two = ia.augmenters.Sometimes(0.25, ia.augmenters.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, shear=(-3, 3)))
    three = ia.augmenters.Sometimes(0.55, ia.augmenters.Affine(
            rotate=(-45, 45)))
    augmented = flipH(images=images)
    augmented = flipV(images=augmented)
    augmented = one(images=augmented)
    augmented = two(images=augmented)
    augmented = three(images=augmented)

    for img, i in zip(augmented, range(len(augmented))):
        mask(img)
        #ia.imshow(img)
        save = Image.fromarray(img.astype('uint8'), 'RGB')
        save.save(r"augmented/" + labels[i] + str(i) + ".png", "PNG")
        #imageio.imwrite(r"augmented/" + labels[i] + str(i), img)

    # Write labels to csv
    pd.DataFrame({'label': labels}).to_csv("auglabels.csv")



class Data:
    
    # Maps each class of white blood cell to an integer label
    class_map = {0 : "EOSINOPHIL", 1 : "LYMPHOCYTE", 2 : "MONOCYTE", 3 : "NEUTROPHIL", 4 : "BASOPHIL",
                 "EOSINOPHIL" : 0, "LYMPHOCYTE" : 1, "MONOCYTE" : 2, "NEUTROPHIL" : 3, "BASOPHIL" : 4 }


    # The image data. Each entry is a matrix of shape (480, 640, 3) or (480, 640, 1) for grayscale
    # Converted to a numpy array after the constructor is called
    X = []

    # The labels (vector of length = len(X))
    # Converted to a numpy array after the constructor is called
    Y = []

    def __init__(self, flatten=False, grayscale=False):
        """
        Read in the augmented data which is ready for models
        """
        augDir = r"augmented"
        csv_labels = pd.read_csv("auglabels.csv")["label"].tolist()
        i = 0
        for im in os.listdir(augDir):
            image = imageio.imread(augDir + "/" + im)

            self.X.append(image)
            # Get the integer mapping of this label
            label_as_int = self.class_map[csv_labels[i]]
            self.Y.append(label_as_int)
            i += 1
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        return


