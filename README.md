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

## Step0. Convert .amr(or others) file into .wav
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

## Step1. Split Frame from a wave file
It will split frames automatically when it detects the sound. <br>
Detection standard: dBFS > -15 (max dBFS is 0), and trim sound interval = [sec-50, sec+50]
```sh
python3 editAudio.py -a -i <inputAudioFile> -p <outputAudioPath>
```

Take this command as an example,
```sh
python3 editAudio.py -a -i soundfile/0715_pat_correct.wav -p real_training_data/correct
```

## Step2. Create Spectrogram by particular Fourier Transform Method
* Create Spectrogram
```sh
python3 spectrogram.py -T <Transform> -i <inputAudioPath> -o <saveImagePath>
```
Take this command as an example,
```sh
python3 spectrogram.py -T stft -i 0727_data/testing/ -o 0727_data/testing/
python3 spectrogram.py -T stft -i real_training_data/incorrect/ -o real_training_data/incorrect/img
```
Output
```
Input File ：  sound/
Output Path：  img/stft/

STFT Spectrogram Has Created!
```

## Step3. CNN Training
```sh
python3 cnn.py -i <CorrectImagePath> <InCorrectImagePath>
```
Take this command as an example,
```sh
python3 cnn.py -i ./real_training_data/
```

or Binary Classification
```sh
usage: python3 binary_cnn.py --correct <CorrectImagePath> --incorrect <CorrectImagePath>
```
Example
```sh
python3 binary_cnn.py --correct 0727_data/training/img/stft/ --incorrect 0727_data/incorrect/img/stft/
```

## Step4. Validation
Validate a Audio file
```sh
usage: python3 validation.py -i <AudioFile>
```
Take this command as an example,
```sh
python3 validation.py -i ./0727_data/testing/frame_0.wav
```

or Validate numerous Audio files 
```sh
usage: python3 test.py --path0 <IncorrectAudioPath> --path1 <CorrectAudioPath>
```
Take this command as an example,
```sh
python3 validation.py --path0 ./0727_data/incorrect/ --path1 ./0727_data/validation/
```

## Others

## Step0 to Step4
```sh
$ bash run.sh
```

### Wave plot
```sh
usage: python3 wave_plot.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>
```
Take this command as an example,
```
python3 wave_plot.py -r 1:5 -i sound_data/0727_record/1.wav -o example.png
```