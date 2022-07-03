import tensorflow
import matplotlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from aibro import Training 


class Keras:
    def __init__(self, train, train_l, optimizer_f='adam', loss_f='binary_crossentropy', metric='accuracy'):

        # Change each value of the array to float
        self.train = train.astype('float32')
        #self.valid = valid.astype('float32')

        # Change the labels from integer to categorical data
        #self.cat_train_l = to_categorical(train_l)
        #self.cat_valid_l = to_categorical(valid_l)

        lb = LabelBinarizer()
        labels = lb.fit_transform(train_l)
        self.cat_train_l = to_categorical(labels)

        #labels = lb.fit_transform(valid_l)
        #self.cat_valid_l = to_categorical(labels)
        
        self.NN = Sequential()
        self.__structure__()
        self.NN.compile(optimizer=optimizer_f, loss=loss_f, metrics=[metric])

    def model_info(self):
        print(self.NN.summary())

    def __structure__(self):

        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(256, 256, 1), activation='relu'))
        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))

        self.NN.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))

        self.NN.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))
        
        self.NN.add(layers.Flatten())
        self.NN.add(Dense(512))
        self.NN.add(Dropout(0.5))
        self.NN.add(Dense(2, activation='sigmoid'))


    def train_network(self, batch=32, iteration=100, verb=1):
        from aibro import Training 
        #self.trained = Training.online_fit(model, train_X, train_Y, machine_id= ['p3.16xlarge.od'])
        #self.trained = Training.online_fit(self.train, self.cat_train_l, machine_id= ['p3.16xlarge.od'])
        generator = ImageDataGenerator(
            horizontal_flip=False,
            vertical_flip=False,
        )

        self.trained = self.NN.fit(
                generator.flow(self.train, self.cat_train_l, batch_size=1024),
                #[self.cat_train_l[:, 1], self.cat_train_l[:, 2]],
                #self.cat_train_l, 
                batch_size=batch, 
                epochs=iteration, 
                verbose=verb, 
                workers=5
                #validation_data=(self.valid, self.cat_valid_l)
        )

    def train_network_with_augmentation(self, batch=32, iteration=100, seed_=27, shuffle_=False, verb=1):

        generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
            rotation_range=10,
            shear_range=0.2,
            brightness_range=(0.2, 1.8),
            rescale=1. / 255
        )

        self.trained = self.NN.fit(
                generator.flow(self.train, self.cat_train_l, batch_size=32, seed=seed_, shuffle=shuffle_),
                batch_size=batch, 
                epochs=iteration, 
                verbose=verb, 
                validation_data=generator.flow(self.valid, self.cat_valid_l, batch_size=32, seed=seed_, shuffle=shuffle_)
        )

    def get_report(self):
        report = dict()

        # To get a prediction value
        prediction = self.NN.predict(self.valid)    
        report.update( { 'prediction' : prediction } )

        # To get the loss and accuracy values
        [loss, accuracy] = self.NN.evaluate(self.valid, self.cat_valid_l)
        report.update( { 'loss' : loss } )
        report.update( { 'accuracy' : accuracy } )

        # To get a confusion matrix
        conf_matrix = confusion_matrix(self.cat_valid_l.argmax(axis=1), prediction.argmax(axis=1))
        report.update( { 'confusion matrix' : conf_matrix } )

        # To get a classification report
        class_report = classification_report(self.cat_valid_l.argmax(axis=1), prediction.argmax(axis=1))
        report.update( { 'classification report' : class_report } )

        # [prediction, loss, accuracy, confusion matrix, classification report]
        return report

    def get_NN(self):
        return self.NN

    def get_trained(self):
        return self.trained

    def plot(self, value):
        plt.plot(self.trained.history[value])
        plt.plot(self.trained.history['val_' + value])
        title_ = 'model ' + value
        plt.title(title_)
        plt.ylabel(value)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

