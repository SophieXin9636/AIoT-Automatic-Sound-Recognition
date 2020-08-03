import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import keras.losses
import os, sys, getopt 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

label_dict = {"incorrect":0, "correct":1}

correct_img_path = "./0727_data/training/img/stft/"
incorrect_img_path = "./0727_data/incorrect/img/stft/"
correct_np_path = correct_img_path + "np/"
incorrect_np_path = incorrect_img_path + "np/"

def save_data_npy(imgPath, npPath):
    file_list = os.listdir(imgPath)

    # Load Image, and save to npy
    if( len(file_list)-1 == len(os.listdir(npPath))):
        return;

    for i, filename in enumerate(file_list):
        file = imgPath + filename
        if os.path.isdir(file):
            continue
        img_data = np.array(Image.open(file).convert('RGB')) # Image.open(cur_file).convert('L')
        img_data = np.expand_dims(img_data, axis=0)
        np.save(npPath + str(i), img_data)

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation]) # validation : test
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()

def cnn():
    # Get Correct labels
    save_data_npy(correct_img_path, correct_np_path)
    correct_labels = os.listdir(correct_np_path)

    # Getting first arrays
    X = np.load(correct_np_path + correct_labels[0])
    cnn_shape = X.shape[1:]
    y = np.zeros(X.shape[0])
    y[0] = label_dict["correct"]

    # Append all of the correct dataset into one single array, same goes for y
    for i, label in enumerate(correct_labels[1:]):
        x = np.load(correct_np_path + label)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=label_dict["correct"]))
    assert X.shape[0] == len(y)

    ## Get Incorrect labels
    save_data_npy(incorrect_img_path, incorrect_np_path)
    incorrect_labels = os.listdir(incorrect_np_path)

    # Append all of the correct dataset into one single array, same goes for y
    for i, label in enumerate(incorrect_labels):
        x = np.load(incorrect_np_path + label)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=label_dict["incorrect"]))
    assert X.shape[0] == len(y)

    # loading data
    X_train, X_test, y_label_train, y_label_test = train_test_split(X, y, test_size= (1 - 0.6), random_state=48, shuffle=True)

    # feature normalize
    X_train_normalize = X_train / 255
    X_test_normalize = X_test / 255

    # one-hot encoding
    y_label_train_hot = to_categorical(y_label_train)
    y_label_test_hot  = to_categorical(y_label_test)

    # create Sequential model
    model = Sequential()

    model.add(Conv2D(filters=16, 
                     kernel_size=(5, 5), 
                     padding='same', 
                     input_shape = cnn_shape, 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=36, 
                     kernel_size=(5, 5), 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten
    model.add(Flatten())

    # FNN layer: 128 output
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(2, activation='softmax'))

    print(model.summary())

    # Compile: choose loss function、optimize method
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # training
    train_history = model.fit(X_train_normalize, y_label_train_hot, 
                              batch_size=20, 
                              epochs=10, 
                              verbose=1, 
                              validation_data=(X_test_normalize, y_label_test_hot))

    # save model
    model.save('ASR.h5')  # creates a HDF5 file 'model.h5'

    # Display loss function, training result
    score = model.evaluate(X_test_normalize, y_label_test_hot, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # show training history
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')

def main(argv):
    global correct_img_path, incorrect_img_path
    global correct_np_path, incorrect_np_path

    try:
        opts, args = getopt.getopt(argv, "h", ["correct=", "incorrect="])
    except getopt.GetoptError:
        print('usage: python3 binary_cnn.py --correct <CorrectImagePath> --incorrect <CorrectImagePath>')
        sys.exit(2)
    else:
        for opt, arg in opts:
            if opt == '-h':
                print('usage: python3 binary_cnn.py --correct <CorrectImagePath> --incorrect <CorrectImagePath>')
                sys.exit()
            elif opt == "--correct":
                if arg[-1] != '/':
                    correct_img_path = arg + '/'
                else:
                    correct_img_path = arg
            elif opt == "--incorrect":
                if arg[-1] != '/':
                    incorrect_img_path = arg + '/'
                else:
                    incorrect_img_path = arg
        correct_np_path = correct_img_path + "np/"
        incorrect_np_path = incorrect_img_path + "np/"
        os.makedirs(correct_np_path, exist_ok=True)
        os.makedirs(incorrect_np_path, exist_ok=True)
        print('\nInput Correct   Data Path： ', correct_img_path)
        print('Input Incorrect Data Path： ', incorrect_np_path)
        print('Numpy Correct   Data Save Path: ', correct_np_path)
        print('Numpy Incorrect Data Save Path: ', incorrect_np_path)
        cnn()
        print("\nCNN Training Has Done!\n")

if __name__ == '__main__':
    if sys.argv.__len__() > 1:
        main(sys.argv[1:])
    else:
        print('usage: python3 binary_cnn.py --correct <CorrectImagePath> --incorrect <CorrectImagePath>')