# Sound Analysis

By Sophie Shin

## Introduction Analysis Step
<img src="./img/step.png">

## Install the dependencies
```sh
pip3 install -r requirements.txt
sudo apt-get install ffmpeg
```
or
```sh
sudo bash env.sh
```

## Usage

### Convert .amr(or others) file into .wav
*Method 1* Only Convert file extension
```sh
ffmpeg -i sound3.amr -ar 22050 sound3.wav
```

*Method 2* Convert file extension and edit particular seconds of Audio
```sh
python3 editAudio.py -r <beginSec>:<endSec> -i <inputAudioFile> -o <outputAudioFile>
```
Take this command as an example,
```sh
python3 editAudio.py -r 5:20 -i soundfile/0601/30s/sound1.wav -o s.wav
```

### Create Spectrogram by particular Fourier Transform Method
* Create Spectrogram
```sh
python3 spectrogram.py -T <Transform> -i <inputAudioPath> -o <saveImagePath>
```
Take this command as an example,
```sh
python3 spectrogram.py -T stft -i sound -o img
```
Output
```
Input File ：  sound/
Output Path：  img/stft/

STFT Spectrogram Has Created!
```

### CNN Training
```sh
python3 mycnn.py
```