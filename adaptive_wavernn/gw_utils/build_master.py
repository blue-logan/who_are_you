from __future__ import print_function
import pandas as pd
import os, sys, glob

from config import config




# print(sys.argv)
# sys.exit()

def savedf(df, filename):
	df.to_csv(filename, index=False)


def combine_dafaframes(frame_list):
	#concats a list of frames
	return pd.concat(frame_list)



if len(sys.argv) == 0:
	praat_regex = config['datadir'] + '/*/' + 'praat.csv'
	phone_regex = config['datadir'] + '/*/' + 'acoustic.csv'
	voice_regex = config['datadir'] + '/*/' + 'voice.csv'
else:
	print('WORKING ON CELEBRITY DATA')
	praat_regex = config['celebrity_datadir'] + '/*/' + 'praat.csv'
	phone_regex = config['celebrity_datadir'] + '/*/' + 'acoustic.csv'
	voice_regex = config['celebrity_datadir'] + '/*/' + 'voice.csv'


praat_frames = [pd.read_csv(path) for path in glob.glob(praat_regex)]
phone_frames = [pd.read_csv(path) for path in glob.glob(phone_regex)]
voice_frames = [pd.read_csv(path) for path in glob.glob(voice_regex)]


if len(sys.argv) == 0:
	savedf(combine_dafaframes(praat_frames), 'master_praat.csv')
	savedf(combine_dafaframes(phone_frames), 'master_acoustic.csv')
	savedf(combine_dafaframes(voice_frames), 'master_voice.csv')

else:
	savedf(combine_dafaframes(praat_frames), config['celebrity_datadir'] + '/' + 'master_praat.csv')
	savedf(combine_dafaframes(phone_frames), config['celebrity_datadir'] + '/' + 'master_acoustic.csv')
	savedf(combine_dafaframes(voice_frames), config['celebrity_datadir'] + '/' + 'master_voice.csv')