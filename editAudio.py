#!/usr/bin/python3
import getopt
import numpy as np
import sys
from pydub import AudioSegment
from pydub.playback import play

#save_path = "./soundfile/0601/10s/"
#num = 1
#filename ="sound"+ str(num) +".wav"
inputfile  = ""
outputfile = ""

def split_sound(start, end):
	# s convert into ms
	s = start * 1000 
	e = end * 1000

	# inport
	file_extension = inputfile.split('.')[1] # wav gpp ... etc
	soundIn = AudioSegment.from_file(inputfile, format=file_extension)
	soundOut = soundIn[s:e] # get s~e msec voice data

	# export sound
	file_prefix = inputfile.split('.')[0]
	##save_filename = file_prefix + "_" + str(start) + ".wav"
	file_extension = outputfile.split('.')[1]
	filename = outputfile.split('.')[0]
	file = soundOut.export(outputfile, format=file_extension)

def main(argv):
	global inputfile
	global outputfile
	begin, end = 0, 0

	try:
		opts, args = getopt.getopt(argv,"hi:o:r:",["ifile=","ofile=","range="])
	except getopt.GetoptError:
		print ('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt == '-h':
				print ('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
				sys.exit()
			elif opt in ("-r", "--range"):
				begin = int(arg.split(':')[0])
				end   = int(arg.split(':')[1])
			elif opt in ("-i", "--ifile"):
				inputfile = arg
			elif opt in ("-o", "--ofile"):
				outputfile = arg
		print ('Input File：', inputfile)
		print ('Output File：', outputfile)
		split_sound(begin, end)

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
	#for num in range(1,4):
	#	filename ="sound"+ str(num) +".wav"
	#	split_sound(20, 30)
	if (sys.argv.__len__()) > 1:
		main(sys.argv[1:])
	else:
		print ('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')

# Reference
# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/365411/