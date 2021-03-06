import os
import sys
from preprocessing import Preprocessing
from processing import Processing
from keras_model import Keras
import tensorflow as tf
import numpy as np


#os.environ["CUDA_VISIBLE_DEVICES"] = '0' #Specify that the first GPU is available
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # The program can only occupy up to 50% of the video memory of the specified gpu
#config.gpu_options.allow_growth = True #The program applies for memory on demand
#sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def data_preprocessing(datadir):
    dataset = Preprocessing(datadir)

    # To get raw data
    train = dataset.getTrain();
    test = dataset.getTest();

    return (train, test)

def data_processing(train, test, fH, fW, fch):
    # To prepare a data for a process of the training
    train_dataset = Processing(train)
    test_dataset = Processing(test)

    f_split = [200, 400, 400]
    s_split = [fH, fW]

    # To normalize a train data and labels
    train_dataset.data_normalization(f_split[0], f_split[1], f_split[2], s_split[0], s_split[1])
    train_data = train_dataset.getData()
    train_labels = train_dataset.getLabels()

    # To normalize a test data and labels
    test_dataset.data_normalization(f_split[0], f_split[1], f_split[2], s_split[0], s_split[1])
    test_data = test_dataset.getData()
    test_labels = test_dataset.getLabels()

    reshaping = [fH, fW, fch]
    train_data = Processing.reshape_data(train_data, reshaping[0], reshaping[1], reshaping[2])
    test_data = Processing.reshape_data(test_data, reshaping[0], reshaping[1], reshaping[2])

    #train_labels = Processing.reshape_data(train_labels, 256, 256, 1)
    #test_labels = Processing.reshape_data(test_labels, 256, 256, 1)

    return (train_data, train_labels, test_data, test_labels)

def data_info(name, data, labels):
    print(f'{name} data: {len(data)} {data[0].shape}')
    print(f'{name} labels: {len(labels)} {set(labels)}')

def main():
    data_train, data_test = data_preprocessing(sys.argv[1])
    (train, train_l, test, test_l) = data_processing(data_train, data_test, 64, 64, 3)

    data_info('Train', train, train_l)
    data_info('Test', test, test_l)

    NN = Keras('VGG16', input_shape=(64, 64, 3))
    NN.data_preparation(train, train_l, test, test_l)
    NN.train_network(batch=256, iteration=20, verb=1)

    report = NN.get_report()

    print('Accuracy:', report['accuracy'])
    print(report['confusion matrix'])
    print(report['classification report'])

if __name__ == '__main__':
    main()

