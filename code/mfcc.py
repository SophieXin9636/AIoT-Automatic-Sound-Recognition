import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import sklearn 

y1, sr1 = librosa.load("./soundfile/1.wav",sr=None)
y2, sr2 = librosa.load("./soundfile/2.wav",sr=None)

mfccs1 = librosa.feature.mfcc(y1, sr=sr1)
print(mfccs1.shape)
mfccs2 = librosa.feature.mfcc(y2, sr=sr2)
print(mfccs2.shape)

fig = plt.figure()
plt.subplot(2, 2, 1)
#Displaying the MFCCs: 
librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')
plt.title('Incorrect Percussion')

plt.subplot(2, 2, 2)
#Displaying the MFCCs: 
librosa.display.specshow(mfccs2, sr=sr2, x_axis='time')
plt.title('Correct Percussion')

plt.subplot(2, 2, 3)
# feature normalization
mfccs1 = sklearn.preprocessing.scale(mfccs1, axis=1) 
librosa.display.specshow(mfccs1, sr=sr1, x_axis='time')

plt.subplot(2, 2, 4)
# feature normalization
mfccs2 = sklearn.preprocessing.scale(mfccs2, axis=1) 
librosa.display.specshow(mfccs2, sr=sr2, x_axis='time')

plt.show()
fig.savefig("./img/mfcc.png")