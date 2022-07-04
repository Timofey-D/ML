import os
import string
from itertools import repeat
#from sklearn.datasets import load_files


class Preprocessing:
   
    def __init__(self, dirdata):

        self.x_labels = []
        self.y_labels = []
        self.train = []
        self.test = []

        self.full_train_path = None
        self.full_test_path = None

        self.dataset_path = self.__check_path__(dirdata)
        self.__split_train_and_test__()

    def __split_train_and_test__(self):

        groups = os.listdir(self.dataset_path)
        groups.sort()

        for group in groups:
            group_path = os.path.join(self.dataset_path, group)
            group_content = os.listdir(group_path)
            
            for datadir in group_content:
                image_path = os.path.join(group_path, datadir)
                
                label = image_path.split(os.sep)[-2]
                try:
                    number_of_images = len(os.listdir(image_path))
                except NotADirectoryError:
                    pass

                if datadir == 'train':
                    self.x_labels.extend(repeat(label, number_of_images))
                    self.train += [os.path.join(image_path, os.listdir(image_path)[ind]) for ind in range(100)]
                elif datadir == 'test':
                    self.y_labels.extend(repeat(label, number_of_images))
                    self.test += [os.path.join(image_path, os.listdir(image_path)[ind]) for ind in range(100)]

    def __check_path__(self, dirdata):
        path = None

        try:
            # To get a path to dataset
            path = os.path.join(os.getcwd(), dirdata)
        except:
            raise Exception("The directory does not exist or the command was entered wrong!")

        return path

    def getXlabels(self):
        """
            The method returns the train labels of the group 
        """
        return self.x_labels

    def getYlabels(self):
        """
            The method returns the test labels of the group 
        """
        return self.y_labels

    def getTrain(self):
        """
            The method returns the train data of the group 
        """
        return self.train

    def getTest(self):
        """
            The method returns the test data of the group 
        """
        return self.test

    def getDatasetPath(self):
        """
            The method returns the path towards to images
        """
        return self.dataset_path

