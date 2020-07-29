import os, sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

inputfile  = "test.wav"
outputfile = "wave.png"

def create_wave_pic(start, end):
	global inputfile, outputfile
	print('Input  File：', inputfile)
	print('Output File：', outputfile)
	# convert into ms unit
	s = start * 1000 
	e = end * 1000

	# inport
	file_extension = inputfile.split('.')[1] # wav gpp ... etc
	soundIn = AudioSegment.from_file(inputfile, format=file_extension)
	sd = soundIn[s:e] # get s~e msec voice data

	t = []
	amp = []
	for i in range(len(sd)):
		t.append(i)
		amp.append(sd[i].dBFS)

	fig = plt.figure()
	plt.xlabel("Time (ms)")
	plt.ylabel("Amplitude (dBFS)")
	plt.plot(t,amp)
	plt.show()
	fig.savefig(outputfile)
	plt.close('all')

def main(argv):
	global inputfile, outputfile
	begin, end = 0, 0

	try:
		opts, args = getopt.getopt(argv,"hi:o:r:",["help","ifile=","ofile=","range="])
	except getopt.GetoptError:
		print ('usage: python3 wave_plot.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
		sys.exit(2)
	else:
		for opt, arg in opts:
			if opt in ('-h', "--help"):
				print('usage: python3 wave_plot.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
				sys.exit()
			elif opt in ("-r", "--range"):
				begin = int(arg.split(':')[0])
				end   = int(arg.split(':')[1])
			elif opt in ("-i", "--ifile"):
				inputfile = arg
			elif opt in ("-o", "--ofile"):
				outputfile = arg
		create_wave_pic(begin, end)

if( __name__ == '__main__'):
	if (sys.argv.__len__()) > 1:
		main(sys.argv[1:])
	else:
		print('usage: python3 wave_plot.py -r <begin>:<end> -i <inputAudioFile> -o <outputAudioFile>')
