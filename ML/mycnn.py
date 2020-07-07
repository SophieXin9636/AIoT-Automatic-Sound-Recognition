import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import Image
import os

extension = ".png"
sec_len = 30
dict_labels = {"incorrect":0, "correct":1}
save_path = ""

def save_data_npy(filename):
    # Load Image, and save to npy
    cur_file = img_path + filename + extension
    cqt_img = np.array(Image.open(cur_file).convert('RGB')) # Image.open(cur_file).convert('L')
    cqt_img = np.expand_dims(cqt_img, axis=0)
    np.save(save_path + filename, cqt_img)
    #print(cqt_img.shape)

def get_labels(path=save_path):
    # Input: Folder Path
    # Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
    labels = os.listdir(path)
    label_indices = np.arange(1, len(labels))
    # to_categorical: one hot encoding
    return labels, label_indices, to_categorical(label_indices)

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation]) # validation : test
    plt.title('Train History')  
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 
    

# save correct data to .npy
save_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/correct/stft_abs/np/"
img_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/correct/stft_abs/"

for sec in range(1,sec_len): # sec
    save_data_npy("1_"+str(sec))

# save incorrect data to .npy
save_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/incorrect/stft_abs/np/"
img_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/incorrect/stft_abs/"

for sec in range(1,sec_len): # sec
    save_data_npy("2_"+str(sec))
    
save_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/correct/stft_abs/np/"
# Get available labels
labels, indices, _ = get_labels(save_path)

# Getting first arrays
X = np.load(save_path + labels[0])
y = np.zeros(X.shape[0])
y[0] = dict_labels["correct"]

# Append all of the correct dataset into one single array, same goes for y
for i, label in enumerate(labels[1:]):
    x = np.load(save_path + label)
    X = np.vstack((X, x))
    y = np.append(y, np.full(x.shape[0], fill_value=dict_labels["correct"]))
assert X.shape[0] == len(y)

save_path = "/content/drive/My Drive/Colab Notebooks/soundAnalysis/training_data/incorrect/stft_abs/np/"
labels, indices, _ = get_labels(save_path)

# Append all of the incorrect dataset into one single array, same goes for y
for i, label in enumerate(labels):
    x = np.load(save_path + label)
    X = np.vstack((X, x))
    y = np.append(y, np.full(x.shape[0], fill_value=dict_labels["incorrect"]))
print(X.shape[0], len(y))
assert X.shape[0] == len(y)

# loading data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= (1 - 0.6), random_state=48, shuffle=True)
#X_train = X_train.reshape(X_train.shape[0], x.shape[0], x.shape[1], x.shape[2])
#X_test = X_test.reshape(X_test.shape[0], x.shape[0], x.shape[1], x.shape[2])

# feature normalize
X_train_normalize = X_train / 255
X_test_normalize = X_test / 255

# one-hot encoding
y_train_hot = to_categorical(y_train)
y_test_hot  = to_categorical(y_test)

# create Sequential model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape = (288, 384, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten
model.add(Flatten())

# FNN layer: 128 output
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Add output layer
model.add(Dense(2, activation='softmax'))

print(model.summary())

# Compile: choose loss function、optimize method
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# training, 訓練過程會存在 train_history
train_history = model.fit(X_train_normalize, y_train_hot, batch_size=10, epochs=100, verbose=1, validation_data=(X_test_normalize, y_test_hot))

# save model
from keras.models import load_model
model.save('ASR.h5')  # creates a HDF5 file 'model.h5'

# 顯示損失函數、訓練成果(分數)
score = model.evaluate(X_test_normalize, y_test_hot, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# show training history
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')