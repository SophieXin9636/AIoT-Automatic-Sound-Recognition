import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display


filename = "sound1.wav" #+ input("input File name: ")
y, sr = librosa.load(filename, sr=None)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)
plt.show()