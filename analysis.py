import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import sklearn 


def analysis(num):
	#num = 1
	source_path = "./soundfile/0601/5s/sound"
	filename = source_path + str(num) +".wav"
	y1, sr1 = librosa.load(filename, sr=None)
	save_path = "./img/0601/5s/"

	# MFCC
	mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
	#print(mfccs1.shape)
	fig = plt.figure()
	librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')
	plt.title('Incorrect Percussion')
	plt.tight_layout()
	fig.savefig(save_path + "mfcc"+ str(num) +".png")
	plt.clf()

	# MFCC feature normalization
	mfccs1 = sklearn.preprocessing.scale(mfccs1, axis=1) 
	librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')
	plt.tight_layout()
	fig.savefig(save_path + "mfcc_normal"+ str(num) +".png")
	plt.clf()

	# CQT
	fig = plt.figure()
	CQT1 = librosa.amplitude_to_db(librosa.cqt(y1, sr=sr1), ref=np.max)
	librosa.display.specshow(CQT1, y_axis='cqt_note')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Constant-Q power spectrogram (note)')
	plt.tight_layout()
	fig.savefig(save_path + 'cqt'+ str(num) +'.png')
	plt.clf()

	## stft
	fig = plt.figure()
	D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
	librosa.display.specshow(D, y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Linear-frequency power spectrogram')
	plt.tight_layout()
	fig.savefig(save_path + "stft"+ str(num) +".png")
	plt.clf()

	## stft
	fig = plt.figure()
	X1 = librosa.stft(y1)
	Xdb1 = librosa.amplitude_to_db(abs(X1))
	librosa.display.specshow(Xdb1, x_axis='time', y_axis='hz')
	plt.colorbar()
	plt.tight_layout()
	fig.savefig(save_path + "stft_abs"+ str(num) +".png")
	plt.clf()

	## stft
	# log
	fig = plt.figure()
	librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='log')
	plt.colorbar()
	plt.tight_layout()
	fig.savefig(save_path + "stft_log"+ str(num) +".png")
	plt.clf()

if( __name__ == '__main__'):
	for i in range(1,4):
		analysis(i)