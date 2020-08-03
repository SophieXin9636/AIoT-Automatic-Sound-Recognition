import librosa
import librosa.display
import os, sys, getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pydub import AudioSegment
from keras.models import load_model
from keras.utils import to_categorical

label_dict = {"incorrect":0, "correct":1}
# open wave file (assume it is 64 ms frame)
my_dpi = 24
x_test = None
y_test_label = None
input_label = "correct" # default

def sound_recognition(file_or_path):
	global x_test, y_test_label
	add_img_data_into_validation_set(file_or_path)

	# add data (one hot encoding shape[1] is 2)
	"""
	a = np.array(Image.open(file_or_path+"correct.png").convert('RGB'))
	a = np.expand_dims(a, axis=0)
	x_test = np.vstack((x_test, a))
	y_test_label = np.append(y_test_label, np.full(a.shape[0], fill_value=label_dict["correct"]))
	"""

	y_test_label_hot = to_categorical(y_test_label)

	print(x_test.shape) # (379, 116, 80, 3)
	print(y_test_label_hot.shape) # (379, 1) -> (379, 2)

	# Load CNN trained model
	model = load_model("ASR.h5")
	prediction = model.predict_classes(x_test)
	#print("Predict: ", prediction)
	score = model.evaluate(x_test, y_test_label_hot, verbose=0)
	# Output result
	print('\nTest loss:', score[0])
	print('Test accuracy:', score[1])

	# show confusion matrix
	print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))

def add_img_data_into_validation_set(file_or_path):
	global x_test, y_test_label

	file_list = []
	if os.path.isdir(file_or_path):
		file_list = [f for f in os.listdir(file_or_path) if os.path.isfile(file_or_path+f)]
		input_path = file_or_path
	else:
		file_num = 1
		file_list.append(file_or_path)
		# parse file path
		path_idx = file.find('/', 2)
		if path_idx > 0:
			while path_idx >= 0:
				pre_idx = path_idx
				path_idx = file.find('/', path_idx+1)
			input_path = file[:pre_idx+1]

	file_num = len(file_list)

	fig = plt.figure(dpi=my_dpi, figsize=(4,6), frameon=False)

	for i, file in enumerate(file_list):
		f = input_path+file
		if file.split('.')[1] != 'wav':
			continue
		sound = AudioSegment.from_file(f)
		if len(sound) == 64:
			# create spectrogram (wave to STFT)
			y, sr = librosa.load(f, sr=None)
			stft_data = librosa.stft(y)
			Xdb1 = librosa.amplitude_to_db(abs(stft_data))
			librosa.display.specshow(Xdb1)
			fig.savefig(input_path+"test.png", dpi=my_dpi, bbox_inches='tight')
			plt.clf()

			# Load Image data, shape = (116,80,3)
			if file_num != 1:
				if i == 0:
					X = np.array(Image.open(input_path+"test.png").convert('RGB'))
					X = np.expand_dims(X, axis=0)
					y_test_label = np.zeros(X.shape[0])
					y_test_label[0] = label_dict[input_label]
				else:
					x_test = np.array(Image.open(input_path+"test.png").convert('RGB'))
					x_test = np.expand_dims(x_test, axis=0)
					X = np.vstack((X, x_test))
					y_test_label = np.append(y_test_label, np.full(x_test.shape[0], fill_value=label_dict[input_label]))
			else:
				x_test = np.array(Image.open(input_path+"test.png").convert('RGB'))
				x_test = np.expand_dims(x_test, axis=0)
				y_test_label = np.zeros(x_test.shape[0])
				y_test_label[0] = label_dict[input_label]

		else:
			print("This is not standard frame (not 64ms)!\n")
	
	if file_num > 1:
		assert X.shape[0] == len(y_test_label)
		x_test = X

	x_test = x_test / 255 # normalize

def main(argv):
	global input_label

	try:
		opts, args = getopt.getopt(argv, "hi:p:", ["file=","path=","label="])
	except getopt.GetoptError:
		print('usage: python3 validation.py -i <AudioFile>')
		print('python3 test.py -p <inputAudioFile> --label <label>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt == '-h':
				print('usage: python3 validation.py -i <AudioFile>')
				print('python3 test.py -p <inputAudioFile> --label <label>')
				sys.exit()
			elif opt == "--label":
				input_label = arg
			elif opt in ("-i", "--file"):
				file_or_path = arg
				print('\nInput  File: ', file_or_path)
			elif opt in ("-p", "--path"):
				if arg[-1] != '/':
					file_or_path = arg + '/'
				else:
					file_or_path = arg
				print('Input Path: ', file_or_path)

		print("Input label: ", input_label, end="\n\n")
		sound_recognition(file_or_path)
		print('\nSound Validation has Done!')

if __name__ == '__main__':
	if sys.argv.__len__() > 3:
		main(sys.argv[1:])
	else:
		print('usage: python3 validation.py -i <AudioFile>')
		print('python3 test.py -p <inputAudioFile> --label <label>')