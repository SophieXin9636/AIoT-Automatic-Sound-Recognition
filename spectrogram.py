#!/usr/bin/python3
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import sklearn.preprocessing
import os, sys, getopt

# default
method = "stft"
intput_audio_path = './sound/'
save_image_path = './img/'
output_extension = ".png"
my_dpi = 24
figsize = (4,6)
y, sr = None, None
fig = None

def analysis_all(file):
	global y, sr, output_extension
	savefile = file.split('.')[0] + output_extension

	# MFCC
	"""
	mfccs_data = librosa.feature.mfcc(y, sr=sr)
	librosa.display.specshow(mfccs_data, sr=sr)
	fig.savefig(save_image_path + "mfcc/"+ savefile, dpi=my_dpi, bbox_inches='tight')
	plt.clf()
	"""
	# MFCC feature normalization
	mfccs_data = librosa.feature.mfcc(y, sr=sr)
	mfccs1 = sklearn.preprocessing.scale(mfccs_data, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr)
	os.makedirs(save_image_path + "mfcc_normal/", exist_ok=True)
	fig.savefig(save_image_path + "mfcc_normal/"+ savefile, dpi=my_dpi, bbox_inches='tight')
	plt.clf()

	# CQT
	CQT1 = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
	librosa.display.specshow(CQT1)
	os.makedirs(save_image_path + "cqt/", exist_ok=True)
	fig.savefig(save_image_path + "cqt/"+ savefile, dpi=my_dpi, bbox_inches='tight')
	plt.clf()

	## stft
	D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
	librosa.display.specshow(D)
	os.makedirs(save_image_path + "stft/", exist_ok=True)
	fig.savefig(save_image_path + "stft/"+ savefile, dpi=my_dpi, bbox_inches='tight')
	plt.close('all')

def analysis_mfcc():
	global y, sr

	mfccs_data = librosa.feature.mfcc(y, sr=sr)
	librosa.display.specshow(mfccs_data, sr=sr)

def analysis_mfcc_normal():
	global y, sr

	mfccs_data = librosa.feature.mfcc(y, sr=sr)
	mfccs1 = sklearn.preprocessing.scale(mfccs_data, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr)

def analysis_cqt():
	global y, sr

	cqt_data = librosa.cqt(y, sr=sr)
	CQT1 = librosa.amplitude_to_db(cqt_data, ref=np.max)
	librosa.display.specshow(CQT1)

def analysis_stft_abs():
	global y, sr

	stft_data = librosa.stft(y)
	Xdb1 = librosa.amplitude_to_db(abs(stft_data))
	librosa.display.specshow(Xdb1)

def call_Transform(method):
	global intput_audio_path, save_image_path
	global output_extension, fig, y, sr

	audio_files = os.listdir(intput_audio_path)
	fig = plt.figure(dpi=my_dpi, figsize=figsize, frameon=False)

	for file in audio_files:
		filename = intput_audio_path + file
		if os.path.isdir(filename):
			continue
		y, sr = librosa.load(filename, sr=None)
		
		if method == "mfcc":
			analysis_mfcc_normal()
		elif method == "stft":
			analysis_stft_abs()
		elif method == "cqt":
			analysis_cqt()
		elif method == "all":
			analysis_all(file)
			continue

		fig.savefig(save_image_path + file.split('.')[0] + output_extension, dpi=my_dpi, bbox_inches='tight')
		plt.clf()

	plt.close('all')

def main(argv):
	global intput_audio_path, save_image_path, method

	try:
		opts, args = getopt.getopt(argv, "hT:i:o:", ["method=","inpath=","outpath="])
	except getopt.GetoptError:
		print('usage: python3 spectrogram.py -T <Transform> -i <inputAudioPath> -o <saveImagePath>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt == '-h':
				print('usage: python3 spectrogram.py -T <Transform> -i <inputAudioPath> -o <saveImagePath>')
				sys.exit()
			elif opt in ("-T", "--method"):
				method = arg
			elif opt in ("-i", "--inpath"):
				if arg[-1] != '/':
					intput_audio_path = arg + '/'
				else:
					intput_audio_path = arg
				if os.path.exists(arg) == False:
					os.makedirs(arg)
			elif opt in ("-o", "--outpath"):
				if arg[-1] != '/':
					save_image_path = arg + '/img/' + method + '/'
				else:
					save_image_path	= arg + 'img/' + method + '/'
				os.makedirs(save_image_path, exist_ok=True)
		print('Input  File： ', intput_audio_path)
		print('Output Path： ', save_image_path)
		call_Transform(method)
		print("\n"+ method.upper() + " Spectrogram Has Created!")

if( __name__ == '__main__'):
	if (sys.argv.__len__()) > 1:
		main(sys.argv[1:])
	else:
		print('usage: python3 spectrogram.py -T <Transform> -i <inputAudioPath> -o <saveImagePath>')
