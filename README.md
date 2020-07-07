# Sound Analysis
By Sophie Shin

## install
* python3
	* pydub
```shell
sudo apt-get install python3-pip
sudo pip3 install pydub
sudo apt-add-repository ppa:mc3man/trusty-media
sudo apt-get update
```
* ffmpeg
```shell
sudo apt-get install ffmpeg
```

or

```shell
sudo bash env.sh
```

## How to convert .amr file into .wav?
*Method 1*
```shell
ffmpeg -i sound3.amr -ar 22050 sound3.wav
```

*Method 2* RUN *editAudio.py* (Undone)
```shell
python3 editAudio.py *files*
```

## simple analysis
* *analysis1.py*
```shell
python3 analysis1.py
```
* *analysis.py*
```shell
python3 analysis1.py
```

## How to analysis?
* create image
```shell
python3 image_create.py
```
* CNN training
```shell
python3 mycnn.py
```