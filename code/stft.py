import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# y: waveform
# sr: sampling rate
y1, sr1 = librosa.load("./soundfile/1.wav",sr=None)
y2, sr2 = librosa.load("./soundfile/2.wav",sr=None)


fig = plt.figure()
D = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
plt.subplot(1, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


D = librosa.amplitude_to_db(librosa.stft(y2), ref=np.max)
plt.subplot(1, 2, 2)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

plt.show()
fig.savefig('./img/stft.png')

"""
y, sr = librosa.load("0727_data/training/frame_230.wav", sr=16000)
librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(y, n_fft=1024))), x_axis='time', y_axis='linear'); plt.colorbar(format='%+2.0f dB')
plt.show()
"""