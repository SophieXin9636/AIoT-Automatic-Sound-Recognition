import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from keras.models import load_model

# open wave file (assume it is 64 ms frame)
sound = AudioSegment.from_file(inputfile)
my_dpi = 48

if(len(sound) == 64):
	# create spectrogram (wave to STFT)
	y, sr = librosa.load(inputfile, sr=None)
	stft_data = librosa.stft(y)
	Xdb1 = librosa.amplitude_to_db(abs(stft_data))
	fig = plt.figure(dpi=my_dpi, figsize=(4,6)) # 192x288
	librosa.display.specshow(Xdb1)

	# Load CNN training model
	model = load_model("ASR.h5")
	print("Test: ", model.predict())
