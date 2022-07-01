import os
import sys
from preprocessing import Preprocessing
from processing import Processing
from keras_model import Keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def data_preprocessing(datadir):
    dataset = Preprocessing(datadir)

    # To get raw data
    train = dataset.getTrain();
    test = dataset.getTest();

    return (train, test)

def data_processing(train, test):
    # To prepare a data for a process of the training
    train_dataset = Processing(train)
    test_dataset = Processing(test)

    # To normalize a train data and labels
    train_dataset.data_normalization(200, 400, 400)
    train_data = train_dataset.getData()
    train_labels = train_dataset.getLabels()

    # To normalize a test data and labels
    test_dataset.data_normalization(200, 400, 400)
    test_data = test_dataset.getData()
    test_labels = test_dataset.getLabels()

    train_data = Processing.reshape_data(train_data, 256, 256, 1)
    test_data = Processing.reshape_data(test_data, 256, 256, 1)
    #train_labels = Processing.reshape_data(train_labels, 256, 256, 1)
    #test_labels = Processing.reshape_data(test_labels, 256, 256, 1)

    return (train_data, train_labels, test_data, test_labels)

def data_info(name, data, labels):
    print(f'{name} data: {len(data)} {data[0].shape}')
    print(f'{name} labels: {len(labels)} {set(labels)}')

def main():
    data_train, data_test = data_preprocessing(sys.argv[1])
    (train, train_l, test, test_l) = data_processing(data_train, data_test)

    data_info('Train', train, train_l)
    data_info('Test', test, test_l)

    (X, Y, XL, YL) = Processing.split_data(train, train_l, rand_state=0)

    NN = Keras(X, Y, XL, YL)
    NN.train_network(batch=256, iteration=2)

if __name__ == '__main__':
    main()

