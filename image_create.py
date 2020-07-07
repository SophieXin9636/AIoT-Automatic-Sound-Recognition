import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import sklearn 

source_path = "./soundfile/0601/sound"
save_path = "./training_data/614/"
my_dpi = 120

def analysis_all(num, sec):

	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	# MFCC
	"""
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc"+ savefile)
	plt.clf()
	"""
	# MFCC feature normalization
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	mfccs1 = sklearn.preprocessing.scale(mfccs1, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc_normal/"+ savefile)
	plt.clf()

	# CQT
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	CQT1 = librosa.amplitude_to_db(librosa.cqt(y1, sr=sr1), ref=np.max)
	librosa.display.specshow(CQT1)
	fig.savefig(save_path + 'cqt/'+ str(num)+"_"+str(sec) +'.png')
	plt.clf()

	## stft
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
	librosa.display.specshow(D)
	fig.savefig(save_path + "stft/"+ savefile)
	plt.clf()

	## stft abs
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	X1 = librosa.stft(y1)
	Xdb1 = librosa.amplitude_to_db(abs(X1))
	librosa.display.specshow(Xdb1)
	fig.savefig(save_path + "stft_abs/"+ savefile)
	plt.clf()

	## stft (log)
	"""
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	librosa.display.specshow(Xdb1, sr=sr1)
	fig.savefig(save_path + "stft_log/"+ savefile)
	"""
	plt.close('all')

def analysis_mfcc(num, sec):
	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+"_"+str(sec)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc"+ savefile)
	plt.close('all')

def analysis_mfcc_normal(num, sec):
	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+"_"+str(sec)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	mfccs1 = sklearn.preprocessing.scale(mfccs1, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc_normal/"+ savefile)
	plt.close('all')

def analysis_cqt(num, sec):
	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+"_"+str(sec)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	CQT1 = librosa.amplitude_to_db(librosa.cqt(y1, sr=sr1), ref=np.max)
	librosa.display.specshow(CQT1)
	fig.savefig(save_path + 'cqt/'+ str(num)+"_"+str(sec) +'.png')
	plt.close('all')

def analysis_stft(num, sec):
	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+"_"+str(sec)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
	librosa.display.specshow(D)
	fig.savefig(save_path + "stft/"+ savefile)
	plt.close('all')

def analysis_stft_abs(num, sec):
	savefile = str(num)+"_"+str(sec)+".png"
	filename = source_path + str(num)+"_"+str(sec)+".wav" #source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	fig = plt.figure(dpi=my_dpi, figsize=(4,3))
	X1 = librosa.stft(y1)
	Xdb1 = librosa.amplitude_to_db(abs(X1))
	librosa.display.specshow(Xdb1)
	fig.savefig(save_path + "stft_abs/"+ savefile)
	plt.close('all')

"""
if( __name__ == '__main__'):
	for j in range(1,2): # sec
		for i in range(1,3): # file
			if i==1:
				save_path = "./training_data/614/"
			elif i==2:
				save_path = "./training_data/614/"
			elif i==3:
				save_path = "./training_data/614/test/"
			analysis(i, j)
"""

if( __name__ == '__main__'):
	save_path = "/home/ais3/Desktop/Linux/soundAnalysis/training_data/correct/"
	for i in range(1,30):
		analysis_cqt(1, i)