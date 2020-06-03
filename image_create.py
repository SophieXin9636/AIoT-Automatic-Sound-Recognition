import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import sklearn 

save_path = "./training_data/"

def analysis(num, sec):
	#num = 1
	source_path = "./soundfile/0601/sound"
	filename = source_path + str(num)+"_"+str(sec)+".wav"
	y1, sr1 = librosa.load(filename, sr=None)

	# MFCC
	"""
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	#print(mfccs1.shape)
	fig = plt.figure(dpi = 24, figsize=(4,3))
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc"+ str(num)+"_"+str(sec)+".png")
	plt.clf()
	"""
	# MFCC feature normalization
	fig = plt.figure(dpi = 24, figsize=(4,3))
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	mfccs1 = sklearn.preprocessing.scale(mfccs1, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr1)
	fig.savefig(save_path + "mfcc_normal/"+ str(num)+"_"+str(sec)+".png")
	plt.clf()

	# CQT
	fig = plt.figure(dpi = 24, figsize=(4,3))
	CQT1 = librosa.amplitude_to_db(librosa.cqt(y1, sr=sr1), ref=np.max)
	librosa.display.specshow(CQT1)
	fig.savefig(save_path + 'cqt/'+ str(num)+"_"+str(sec) +'.png')
	plt.clf()

	## stft
	fig = plt.figure(dpi = 24, figsize=(4,3))
	D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
	librosa.display.specshow(D)
	fig.savefig(save_path + "stft/"+ str(num)+"_"+str(sec) +".png")
	plt.clf()

	## stft
	fig = plt.figure(dpi = 24, figsize=(4,3))
	X1 = librosa.stft(y1)
	Xdb1 = librosa.amplitude_to_db(abs(X1))
	librosa.display.specshow(Xdb1)
	fig.savefig(save_path + "stft_abs/"+ str(num)+"_"+str(sec) +".png")
	plt.clf()

	## stft (log)
	fig = plt.figure(dpi = 24, figsize=(4,3))
	librosa.display.specshow(Xdb1, sr=sr1)
	fig.savefig(save_path + "stft_log/"+ str(num)+"_"+str(sec) +".png")
	plt.close('all')

if( __name__ == '__main__'):
	for i in range(1,4): # file
		if i==1:
			save_path = "./training_data/correct/"
		elif i==2:
			save_path = "./training_data/incorrect/"
		elif i==3:
			save_path = "./training_data/test/"
		for j in range(1,30): # sec
			analysis(i, j)