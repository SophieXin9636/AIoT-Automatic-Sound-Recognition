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
y_test_label = []
input_label = "correct" # default
correct_path = ""
incorrect_path = ""

def sound_recognition():
	global x_test, y_test_label, correct_path, incorrect_path
	add_img_data_into_validation_set(correct_path, "correct")
	add_img_data_into_validation_set(incorrect_path, "incorrect")
	
	x_test = x_test / 255 # normalize
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
	print('Test accuracy:', score[1], end="\n\n")

	# show confusion matrix
	print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))

def add_img_data_into_validation_set(file_or_path, label):
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
			y, sr = librosa.load(f, sr=16000)
			stft_data = librosa.stft(y)
			Xdb1 = librosa.amplitude_to_db(abs(stft_data))
			librosa.display.specshow(Xdb1)
			fig.savefig(input_path+"test.png", dpi=my_dpi, bbox_inches='tight')
			plt.clf()

			# Load Image data, shape = (116,80,3)
			if file_num > 1:
				if len(y_test_label) == 0:
					x_test = np.array(Image.open(input_path+"test.png").convert('RGB'))
					x_test = np.expand_dims(x_test, axis=0)
					y_test_label = np.zeros(x_test.shape[0])
					y_test_label[0] = label_dict[input_label]
				else:
					data = np.array(Image.open(input_path+"test.png").convert('RGB'))
					data = np.expand_dims(data, axis=0)
					x_test = np.vstack((x_test, data))
					y_test_label = np.append(y_test_label, np.full(data.shape[0], fill_value=label_dict[label]))
			else:
				x_test = np.array(Image.open(input_path+"test.png").convert('RGB'))
				x_test = np.expand_dims(x_test, axis=0)
				y_test_label = np.zeros(x_test.shape[0])
				y_test_label[0] = label_dict[input_label]

		else:
			print("This is not standard frame (not 64ms)!\n")
	
	if file_num > 1:
		assert x_test.shape[0] == len(y_test_label)

def main(argv):
	global correct_path, incorrect_path

	try:
		opts, args = getopt.getopt(argv, "hi:p:", ["file=","path0=","path1="])
	except getopt.GetoptError:
		print('usage: python3 validation.py -i <AudioFile>')
		print('usage: python3 test.py --path0 <IncorrectAudioPath> --path1 <CorrectAudioPath>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt == '-h':
				print('usage: python3 validation.py -i <AudioFile>')
				print('usage: python3 test.py --path0 <IncorrectAudioPath> --path1 <CorrectAudioPath>')
				sys.exit()
			elif opt in ("-i", "--file"):
				file_or_path = arg
				print('\nInput  File: ', file_or_path)
			elif opt == "--path0":
				if arg[-1] != '/':
					incorrect_path = arg + '/'
				else:
					incorrect_path = arg
			elif opt == "--path1":
				if arg[-1] != '/':
					correct_path = arg + '/'
				else:
					correct_path = arg

		print("Input  Correct Path: ", correct_path)
		print("Input Inorrect Path: ", incorrect_path, end="\n\n")
		sound_recognition()
		print('\nSound Validation has Done!')

if __name__ == '__main__':
	if sys.argv.__len__() > 3:
		main(sys.argv[1:])
	else:
		print('usage: python3 validation.py -i <AudioFile>')
		print('usage: python3 test.py --path0 <IncorrectAudioPath> --path1 <CorrectAudioPath>')