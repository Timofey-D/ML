import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import repeat

class Processing:

    def __init__(self, data, labels = None):
        
        self.raw_data = data
        self.data = []
        self.labels = []
        #self.data_normalization()
        
    def getRawData(self):
        """

        """
        return self.raw_data

    def getData(self):
        """

        """
        return self.data

    def getLabels(self):
        """

        """
        return self.labels

    # To split a dataset on several groups
    @staticmethod
    def split_data(data, labels, size_of_test=0.2, rand_state=6, shuffling=True):
        """

        """
        (X, Y, LX, LY) = train_test_split(data, labels, test_size=size_of_test, random_state=rand_state, shuffle=shuffling)
        return (X, Y, LX, LY)

    # To get a pad to image
    def __add_padding__(image):
        """

        """
        (h, w) = image.shape[:2]

        l_side = (h if h > w else w) / 2

        if h > w:
            segment = int(w + l_side)
            image = cv2.copyMakeBorder(image, 0, 0, segment, segment, cv2.BORDER_REPLICATE)
        else:
            segment = int(h + l_side)
            image = cv2.copyMakeBorder(image, segment, segment, 0, 0, cv2.BORDER_REPLICATE)

        return image


    def reshape_data(data, width=32, height=32, channels=1):
        """

        """
        # To reshape data
        data = np.array(data)
        return data.reshape(data.shape[0], width, height, channels)
        

    # To get a normalized image
    def __normalize_image__(image, height, width):
        """

        """
        # To add padding
        p_image = Processing.__add_padding__(image) 

        # To get an image size
        (h, w) = p_image.shape[:2]
        m_side = min(h, w)

        # To center an image
        c_image = p_image[
                int(h / 2 - m_side / 2) : int(h / 2 + m_side / 2), 
                int(w / 2 - m_side / 2) : int(w / 2 + m_side / 2)
                ]

        # To change an image size
        r_image = cv2.resize(c_image, (height,width), cv2.INTER_AREA)

        return r_image

    # To open an image by passed path
    def open_image(image_path):
        """

        """
        try:
            # To open an image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except:
            print("The path:", image_path)
            raise Exception("The file wasn\'t found or the file doesn't exist!")

        return image

    # To print out an image
    def print_image(image):
        """

        """
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # To get a normalized data
    def data_normalization(self, stepSize, fh, fw, sh, sw):
        """

        """
        for path in self.raw_data:
            open_image = Processing.open_image(path)
            pieces = Processing.__split_image__(open_image, stepSize, (fh, fw))

            for piece in pieces:
                image = Processing.__normalize_image__(piece[2], sh, sw)
                self.labels.extend(repeat(path.split(os.sep)[-3], 1))
                self.data.append(image)

            #§if len(self.data) > 36000:
            #    break

        self.data = np.array(self.data)

    def __split_image__(image, stepSize, windowSize):
        """
        """
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
