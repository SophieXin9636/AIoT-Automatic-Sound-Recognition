from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def cnn():
	# loading data
	X = []
	y = []
	base_dir = "./training_data/correct/cqt/"
	for f in os.listdir(base_dir):
	    print(f"{f} is an input image path")
	    X.append(base_dir+f)
	    y.append(base_dir+f)

	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
	image.load_img(X, grayscale=False, color_mode='rgb', target_size=None)

	#X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
	#X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)

	# one-hot encoding
	#y_train_hot = to_categorical(y_train)
	#y_test_hot = to_categorical(y_test)
	#print("X_train.shape=", X_train.shape)




	# create Sequential model
	model = Sequential()
	# convolution layer，filter (output size) = 32
	model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(153, 115, 3)))
	# Pooling layer. and take maximum
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
	model.add(Dropout(0.25))
	# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
	model.add(Flatten())
	# FNN layer: 128 output
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	# Add output layer
	model.add(Dense(3, activation='softmax'))
	# Compile: choose loss function、optimize method
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	# training, 訓練過程會存在 train_history
	model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))


	X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
	X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
	score = model.evaluate(X_test, y_test_hot, verbose=1)

	# 模型存檔
	model.save('ASR.h5')  # creates a HDF5 file 'model.h5'


	# 預測(prediction)
	mfcc = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
	mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
	print("labels=", get_labels())
	print("predict=", np.argmax(model.predict(mfcc_reshaped)))

def testing():
	test_image = image.load_img('./training_data/test/', target_size = (96, 72))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = model.predict(test_image)
	training_set.class_indices
	if result[0][0] == 1:
	  prediction = 'correct'
	else:
	  prediction = 'incorrect'

if( __name__ == '__main__'):
	cnn()