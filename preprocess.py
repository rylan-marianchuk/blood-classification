import pandas as pd
import imgaug as ia
import os
import imageio
import numpy as np
from PIL import Image, ImageOps
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split

class Data:

    """
    Load the image data and augment to 10k observations on the fly
    """

    # Maps each class of white blood cell to an integer label
    class_map = {0 : "EOSINOPHIL", 1 : "LYMPHOCYTE", 2 : "MONOCYTE", 3 : "NEUTROPHIL",
                 "EOSINOPHIL" : 0, "LYMPHOCYTE" : 1, "MONOCYTE" : 2, "NEUTROPHIL" : 3 }
    
    # The image data. Each entry is the filename for a 480x640 image.
    # Converted to a numpy array after the constructor is called
    X = []

    # The labels (vector of length = len(X))
    # Converted to a numpy array after the constructor is called
    Y = []
    
    normalized = False
    def __init__(self, scale, normalize=False, read_from_disk=False, save_to_disk=False):
        """
        Read in the augmented data which is ready for models

        :param scale float how much to resize width and height before augmenting
        :param normalize: if True, values of the image array are floats from 0 to 1, otherwise uint8 between 0, 255
        """
        self.normalized = normalize

        if read_from_disk:
            augDir = r"augmented"
            csv_labels = pd.read_csv("auglabels.csv")["label"].tolist()
            i = 0

            # Loop over image files, ensuring order of filenames matches labels CSV
            # https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
            for im in sorted(os.listdir(augDir), key=lambda f: int(''.join(filter(str.isdigit, f)))):
                # Store image filenames
                self.X.append(im)

                # Get the integer mapping of this image label
                label_as_int = self.class_map[csv_labels[i]]
                self.Y.append(label_as_int)

                #### For counting distribution of blood cell types
                #self.distribution[csv_labels[i]] += 1
                i += 1

        else:
            self.augment(scale, save_to_disk)
            self.X = np.array(self.X)
            self.Y = np.array([self.class_map[s] for s in self.Y])
        
        return
    
    def splitData(self, random_state=None):
        """
            Splits own data (X and y) into training and test sets by
            an 80/20 ratio.
            
            Parameters:
                random_state (optional): a random integer seed for splitting
                
                
            Returns:
                X_train: a Numpy array of filenames for image data
                X_test: a Numpy array of filenames for image data
                y_train: a Numpy array of labels for the data in X_train
                y_test: a Numpy array of labels for the data in X_test
        """

        X_train, X_test, y_train, y_test = train_test_split(
        self.X, self.Y, stratify=self.Y, test_size=0.2, random_state=random_state)
        
        return (X_train, X_test, y_train, y_test)
    
    def extra_processing(images, grayscale=False, flatten=False):
        """
        Processes a provided Numpy array of image vectors according to the
        provided arguments.

        Parameters
        ----------
        images : Numpy array
            Contains RGB image vectors
            
        grayscale: Boolean
            True if images should be converted to grayscale, False otherwise.
            
        flatten: Boolean
            True if the images should be flattened.

        Returns
        -------
        A Numpy array of the images converted to grayscale

        """
        
        rgb_weights = [0.2989, 0.5870, 0.1140]    # For grayscale conversion
        processed_images = []
        
        for index, im in enumerate(images):
            
            # Convert to grayscale and/or flatten image
            if grayscale:
                # https://www.kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python
                im = np.dot(im[...,:3], rgb_weights)
            if flatten:
                im = im.flatten()
                
            processed_images.append(im)
        
        # Change to NP array
        processed_images = np.array(processed_images)
            
        return processed_images

    def isomap_reduce(self, batch, lower_dimensions, consider_n_neighbours=10):
        """
        Apply the isomap dimension reduction on the batch
        :param batch: a subset of the class data self.X. Each entry is an image of 3 RGB channels
        :return: the manifold dimension reduction of the reduced batch
        """
        isomap = Isomap(n_components=lower_dimensions, n_neighbors=consider_n_neighbours)
        return isomap.fit_transform(batch)

    cream_bg = (195, 201, 198)

    # Euclidean distance between two vectors
    def dist(self, vec1, vec2):
        return np.sqrt(sum((vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))))

    def mask(self, img):
        # If all channels are below some threshold, turn to cream
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Mask all but blue
                if self.dist((0, 0, 0), img[i][j]) < 45:
                    img[i][j][0] = self.cream_bg[0]
                    img[i][j][1] = self.cream_bg[1]
                    img[i][j][2] = self.cream_bg[2]
        return


    def augment(self, scale, save_to_disk=False):
        """
        Run on each instance of this class

        Augmentations to apply:
        :param scale float how much to resize width and height before augmenting
        :param save_to_disk: if True, all augmented images are saved, otherwise just loaded into self.X
        :return: augmented dataset
        """

        # How many duplications of non augmented needed
        dups = 33

        # Reading csv for label
        df = pd.read_csv(r"dataset-master/dataset-master/labels.csv")
        label_col = df["Category"].tolist()
        labels = []
        inDIR = r"dataset-master/dataset-master/JPEGImages"
        count = 0
        using = 0
        images = []
        for im in os.listdir(inDIR):
            if type(label_col[count]) == type(float()) or len(label_col[count].split(" ")) != 1 \
                or len(label_col[count].split(",")) != 1 or label_col[count] == 'BASOPHIL' :
                # 0 or more than one label, discard
                # Also discard basophils due to rarity
                count += 1
                continue
            
            # Load and resize images
            imagePIL = Image.open(inDIR + "/" + im)
            imagePIL = imagePIL.resize((int(imagePIL.size[0]*scale), int(imagePIL.size[1]*scale)) , Image.BICUBIC)

            if self.normalized:
                # Center crop
                width, height = imagePIL.size  # Get dimensions
                new_width, new_height = (265, 265)
                left = (width - new_width) / 2
                top = (height - new_height) / 2
                right = (width + new_width) / 2
                bottom = (height + new_height) / 2

                # Crop the center of the image
                imagePIL = imagePIL.crop((left, top, right, bottom))

                # Convert to float 16 to normalize here
                image_as_array = np.array(imagePIL).astype(np.float32) / 255
            else: image_as_array = np.array(imagePIL)
            for _ in range(dups):
                images.append(image_as_array)
                labels.append(label_col[count])
            count += 1
            using += 1
        print("New size:")
        print(images[0].shape)
        print("Number of images dropped: " + str(count - using))
        print("COUNT (number of original images duplicated " + str(dups) + " times: " + str(using))
        print("TOTAL images to create: " + str(using * dups))

        one = ia.augmenters.Sometimes(0.35, ia.augmenters.Affine(
            scale={"x": (0.8, 1.12), "y": (0.8, 1.12)}))
        two = ia.augmenters.Sometimes(0.35, ia.augmenters.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, shear=(-3, 3)))
        three = ia.augmenters.Sometimes(0.55, ia.augmenters.Affine(
            rotate=(-45, 45)))

        aug = ia.augmenters.SomeOf((1, None), [
            ia.augmenters.Affine(rotate=(-45, 45)),
            ia.augmenters.Affine(
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, shear=(-3, 3)),
            ia.augmenters.Affine(
                scale={"x": (0.8, 1.12), "y": (0.8, 1.12)})
        ], random_order=True)

        flipH = ia.augmenters.Fliplr(0.3333)
        flipV = ia.augmenters.Flipud(0.3333)
        augmented = flipH(images=images)
        augmented = flipV(images=augmented)
        augmented = aug(images=augmented)

        if save_to_disk:
            for img, i in zip(augmented, range(len(augmented))):
                self.mask(img)
                # ia.imshow(img)
                save = Image.fromarray(img.astype('uint8'), 'RGB')
                save.save(r"augmented/" + labels[i] + str(i) + ".png", "PNG")
                # imageio.imwrite(r"augmented/" + labels[i] + str(i), img)

            # Write labels to csv
            pd.DataFrame({'label': labels}).to_csv("auglabels.csv")

        self.X = augmented
        self.Y = labels
