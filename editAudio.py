#!/usr/bin/python3
import os, sys, getopt
import numpy as np
from pydub import AudioSegment

inputfile  = ""
outputfile = "test.wav"
outputPath = "./"

def split_sound(start, end):
	global inputfile, outputfile
	print('Input  File：', inputfile)
	print('Output File：', outputfile)
	# s convert into ms
	s = start * 1000 
	e = end * 1000

	# inport
	file_extension = inputfile.split('.')[1] # wav gpp ... etc
	soundIn = AudioSegment.from_file(inputfile, format=file_extension)
	soundOut = soundIn[s:e] # get s~e msec voice data

	# export sound
	file_prefix = inputfile.split('.')[0]
	file_extension = outputfile.split('.')[1]
	filename = outputfile.split('.')[0]
	file = soundOut.export(outputfile, format=file_extension)

def auto_create_frame():
	global inputfile, outputPath
	print('Input  File：', inputfile)
	print('Output Path：', outputPath)

	# inport
	file_extension = inputfile.split('.')[1] # wav gpp ... etc
	sound = AudioSegment.from_file(inputfile, format=file_extension)

	patting_frame_range = []
	i = 0
	# large volume percussion
	while i < int(sound.duration_seconds*1000):
		x = sound[i].dBFS
		if x > -30:
			patting_frame_range.append([i-20, i+180])
			i += 300
		else:
			i += 1

	# export sound
	num_of_file = 0
	files = os.listdir(outputPath)
	print(len(files))
	for f in files:
		if os.path.isfile(outputPath+f):
			num_of_file += 1
	for cnt, data in enumerate(patting_frame_range):
		sound[data[0]:data[1]].export(outputPath+"frame_"+ str(cnt+num_of_file)+".wav", format="wav")
	print("Totally create" , cnt+1, "wave files!")
	print("Output Path：[", outputPath, "] has", len(os.listdir(outputPath)), "wave files")

def main(argv):
	global inputfile, outputfile, outputPath
	begin, end = 0, 0
	mode = "default"

	try:
		opts, args = getopt.getopt(argv,"hai:o:r:p:",["help","ifile=","ofile=","range=","mode=","auto","path="])
	except getopt.GetoptError:
		print ('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt in ('-h', "--help"):
				print('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
				print('       python3 editAudio.py --auto -i <inputAudioFile> -p <outputAudioPath>')
				sys.exit()
			elif opt in ("-r", "--range"):
				begin = int(arg.split(':')[0])
				end   = int(arg.split(':')[1])
			elif opt in ("-i", "--ifile"):
				inputfile = arg
			elif opt in ("-o", "--ofile"):
				outputfile = arg
			elif opt in ("-p", "--path"):
				if arg[-1] != '/':
					outputPath = arg + '/'
				else:
					outputPath = arg
				os.makedirs(outputPath, exist_ok=True)
			elif opt in ("-a", "--auto"):
				mode = "auto"
		if mode == "auto":
			auto_create_frame()
		else:
			split_sound(begin, end)

if( __name__ == '__main__'):
	if (sys.argv.__len__()) > 1:
		main(sys.argv[1:])
	else:
		print('usage: python3 editAudio.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
		print('       python3 editAudio.py -a -i <inputAudioFile> -p <outputAudioPath>')
