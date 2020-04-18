import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

y1, sr1 = librosa.load("./soundfile/1.wav", sr=None)
y2, sr2 = librosa.load("./soundfile/2.wav", sr=None)

fig = plt.figure()
plt.subplot(1, 2, 1)
CQT1 = librosa.amplitude_to_db(librosa.cqt(y1, sr=sr1), ref=np.max)
librosa.display.specshow(CQT1, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

plt.subplot(1, 2, 2)
CQT2 = librosa.amplitude_to_db(librosa.cqt(y2, sr=sr2), ref=np.max)
librosa.display.specshow(CQT2, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

plt.show()
fig.savefig('./img/cqt.png')