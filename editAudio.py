import numpy as np
import sys
from pydub import AudioSegment
from pydub.playback import play

#filelist = sys.argv
#argc = len(sys.argv)
#soundOut1 = []

save_path = "./soundfile/0601/10s/"
num = 1
filename ="sound"+ str(num) +".wav"

def split_sound(start, end):
	# s convert into ms
	s = start * 1000 
	e = end * 1000

	# inport
	file_extension = filename.split('.')[1] # wav gpp ... etc
	soundIn = AudioSegment.from_file(filename, format="wav")
	soundOut = soundIn[s:e] # get s~e msec voice data

	# export sound
	file_prefix = filename.split('.')[0]
	save_filename = file_prefix + "_" + str(start) + ".wav"
	file = soundOut.export(save_path + save_filename, format="wav")


if( __name__ == '__main__'):
	"""
	duration = input("Input sound duration you want to split: ")
	# split sound to ls
	for no in range(1,4): # file num
		num = no
		filename ="sound"+ str(num) +".wav"
		for i in range(1,int(30/duration)): # sec
			split_sound(i, i+duration)
	"""
	for num in range(1,4):
		filename ="sound"+ str(num) +".wav"
		split_sound(20, 30)
# source
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/365411/
