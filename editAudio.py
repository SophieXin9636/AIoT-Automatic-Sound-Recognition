import numpy as np
import sys
from pydub import AudioSegment
from pydub.playback import play

# https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/365411/

# inport
soundIn = AudioSegment.from_file("./7.3gpp", format="3gp")
soundOut1 = soundIn[2000:9000] # get sound1 2~5 sec voice data
soundOut2 = soundIn[9000:13000]

# export
file1 = soundOut1.export("./1.wav", format="wav")
file2 = soundOut2.export("./2.wav", format="wav")