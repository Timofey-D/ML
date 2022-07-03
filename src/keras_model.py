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
from tensorflow.keras.applications import vgg16


class Keras:
    def __init__(self, _type, optimizer_f='adam', loss_f='binary_crossentropy', metric='accuracy', input_shape=(256, 256, 1)):

        self.NN = Sequential()
        if _type == "CNN":
            self.__CNN__(input_shape)
        elif _type == "VGG16":
            self.__VGG16__(input_shape)
        elif _type == "Xception":
            self.__Xception__(input_shape)

        self.NN.compile(optimizer=optimizer_f, loss=loss_f, metrics=[metric])
        

    def model_info(self):
        print(self.NN.summary())

    def __CNN__(self, shape):

        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=shape, activation='relu'))
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
        
    def __VGG16__(self, shape):

        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=shape)
        self.NN.add(vgg_conv)

        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=shape, activation='relu'))
        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        #self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))

        self.NN.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        #self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))

        self.NN.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.NN.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        #self.NN.add(layers.MaxPool2D())
        self.NN.add(Dropout(0.25))
        
        self.NN.add(layers.Flatten())
        self.NN.add(Dense(512))
        self.NN.add(Dropout(0.5))
        self.NN.add(Dense(2, activation='sigmoid'))

    def __Xception__(self, shape=(256, 256, 1)):
        vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=shape)
        model.add(vgg_conv)

        self.NN.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=shape, activation='relu'))
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

    def data_preparation(self, train, train_l, test, test_l):
        # Change each value of the array to float
        self.train = train.astype('float32')
        self.test = test.astype('float32')

        # Change the labels from integer to categorical data
        #self.cat_train_l = to_categorical(train_l)
        #self.cat_valid_l = to_categorical(valid_l)

        lb = LabelBinarizer()
        labels = lb.fit_transform(train_l)
        self.cat_train_l = to_categorical(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(test_l)
        self.cat_test_l = to_categorical(labels)
        #labels = lb.fit_transform(valid_l)
        #self.cat_valid_l = to_categorical(labels)

    def train_network(self, batch=32, iteration=100, verb=1):
        generator = ImageDataGenerator(
            horizontal_flip=False,
            vertical_flip=False,
        )

        self.trained = self.NN.fit(
                generator.flow(self.train, self.cat_train_l, batch_size=64),
                batch_size=batch, 
                epochs=iteration, 
                verbose=verb, 
                workers=5
        )

    def get_report(self):
        report = dict()

        # To get a prediction value
        prediction = self.NN.predict(self.test)    
        report.update( { 'prediction' : prediction } )

        # To get the loss and accuracy values
        [loss, accuracy] = self.NN.evaluate(self.test, self.cat_test_l)
        report.update( { 'loss' : loss } )
        report.update( { 'accuracy' : accuracy } )

        # To get a confusion matrix
        conf_matrix = confusion_matrix(self.cat_test_l.argmax(axis=1), prediction.argmax(axis=1))
        report.update( { 'confusion matrix' : conf_matrix } )

        # To get a classification report
        class_report = classification_report(self.cat_test_l.argmax(axis=1), prediction.argmax(axis=1))
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

