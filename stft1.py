import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

y1, sr1 = librosa.load("./soundfile/1.wav", sr=None)
y2, sr2 = librosa.load("./soundfile/2.wav", sr=None)

fig = plt.figure()
plt.subplot(2, 2, 1)
X1 = librosa.stft(y1)
Xdb1 = librosa.amplitude_to_db(abs(X1))
librosa.display.specshow(Xdb1, x_axis='time', y_axis='hz')
plt.colorbar()

plt.subplot(2, 2, 2)
X2 = librosa.stft(y2)
Xdb2 = librosa.amplitude_to_db(abs(X2))
librosa.display.specshow(Xdb2, x_axis='time', y_axis='hz')
plt.colorbar()

plt.subplot(2, 2, 3)
# log
librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='log')
plt.colorbar()

plt.subplot(2, 2, 4)
# log
librosa.display.specshow(Xdb2, sr=sr2, x_axis='time', y_axis='log')
plt.colorbar()

plt.show()
fig.savefig('./img/stft1.png')