from __future__ import print_function

import glob
import json
import os
import re
from collections import defaultdict

from config import config
# modules made for preprocessing
# This is where the raw files will be process to obtain the data we are looking for
# which includes the praat table (formants), acoustic table (phonemes), bandwidth extractor
# Import any new functionality here
import importlib.util
cache_path = '/home/eric/GuessWho/adaptive_wavernn/gw_utils/preprocess/'

# phonemizer = imp.load_compiled("phomizer", cache_path+'phonemizer.pyc')
spec = importlib.util.spec_from_file_location("phomizer", cache_path+'phonemizer.pyc')
phonemizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phonemizer)

# praatformant = imp.load_compiled("praatformat", cache_path + 'praatformt.pyc')
spec = importlib.util.spec_from_file_location("praatformant", cache_path+'praatformant.pyc')
praatformant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(praatformant)
# from preprocess import phonemizer
# from preprocess import praatformant
from preprocess.bandwith_extractor import *





def changecwd(path):
	#change working dir
	os.chdir(path)

def goto_datadir():
	#goes to data directory
	os.chdir(config['datadir'])

def goto_rootdir():
	# goes to the root directory
	os.chdir(config['rootdir'])

def savedf(df, filename):
	df.to_csv(filename, index=False)


def combine_dafaframes(frame_list):
	#concats a list of frames
	return pd.concat(frame_list)

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def sentence_filenames():
	# print(sorted(glob.glob('*sentence[0-9]*.wav')))
	return glob.glob('*sentence[0-9]*.wav')

def master_sentences():
	master_sentences_dict = defaultdict(str)
	for line in open(config['master_sentence_file'], 'r').readlines():
		number, sentence = line.split(')')
		master_sentences_dict[int(number)] = sentence[1:].strip()
	return master_sentences_dict

def align_phoneme(audiofile, text):
	arpafile = config['arpa_to_ipa_file']
	phone = phonemizer.phoneAligner(audiofile, text, arpafile)

	phone.phonemize()
	phone.parse_results(verbose = False)
	# print(phone.df)
	return phone.df


def extract_formants(audiofile, sex):
	results, stepsize = praatformant.formant_finder(audiofile, sex = sex, remove_output = True)
	df = praatformant.make_dataframe(results)
	return df

def getnumber(filename, type):
	id,sentence = filename.split('_')
	if type == 'id':
		number =  re.findall('\d+',id)[0]
	elif type == 'sentence':
		number = re.findall('\d+',sentence)[0]

	return int(number)

def get_sex(physical , id_number):
	row = physical[physical['id'] == id_number]
	sex =  row['sex'].values[0]
	assert sex in ['m','f']
	return sex


def get_demographics(demographic, key):
	return demographic[key]


def voice_join(df_formants, df_acoustic):
    #Join frames
    new_df = pd.merge(df_formants, df_acoustic,  how='left', on=['id','sentence_number'])
    #Only keep rows where the time from the formant df is in between the start/end time of phoneme
    formatted = new_df[(new_df['time'] > new_df['phoneme_start']) & (new_df['time'] < new_df['phoneme_end'])]
    return formatted

def trim_ends(df, rows_to_remove):
    df['trim_ends'] = True
    df['step_difference'] = df['phoneme_end'].shift(-1) - df['phoneme_end']

    rows_removed = rows_to_remove
    trim = 0
    row_indexes = list(df.index.values)
    for list_index,row_index in enumerate(row_indexes):
        # This trims the start of a phoneme
        if trim < rows_removed:
            df.loc[row_index, 'trim_ends'] = False
            trim += 1

        # if this value is greater than 0, then the next row is the start
        # of another phoneme. This means that we need to start trimming the
        # end
        if df.loc[row_index,'step_difference'] > 0:
            for idx in row_indexes[list_index - rows_removed +1 : list_index+1]:
#                 print(idx)
                df.loc[idx, 'trim_ends'] = False
            trim = 0

    return df

def tag_misprounced(df, sentence_number, word_dict):
	if str(sentence_number) in word_dict.keys():
		list = word_dict[str(sentence_number)]
		df['mispronounced'] = df['word'].isin(word_dict[str(sentence_number)])
	else:
		df['mispronounced'] = False

	return df

if __name__ == '__main__':
	# dict with the key = sentence number, value = sentence itself
	master_sentences_dict = master_sentences()
	
	# Load physical table
	physical_df = pd.read_csv('master_physical.csv')

	#change working directory to data directory
	goto_datadir()

	print(os.getcwd())



	# grabs all subjects in the root data directory
	# A subject is the directory where all the data is stored for *one* individual
	# e.g., root/person1/
	subjects = [subject for subject in os.listdir(config['datadir']) if not subject.startswith('.')]
	
	# Process each subject
	for idx, subject in enumerate(subjects, start=1):

		#output handlers
		phoneme_df_frames = []
		formant_df_frames = []
		voice_df_frames = []

		changecwd(subject) #change working directory to subject's directory
		print('Working on Subject: %s (%d/%d)' % (subject, idx, len(subjects)))


		# Check if the '_PROCESSED' flag is set, if so, move on to next subject
		# no need to look at this for now since we are processing everyone at the
		# same time instead of piecewise
		# if os.path.isfile('_PROCESSED'):
		# 	goto_datadir() # change working directory to data directory
		# 	continue


		# load demographic information for the subject
		demographic = json.load(open('demographic.json','r'))


		# Start processing each sentence for the subject
		for audio_filename in sorted(sentence_filenames()):
			# --- Start processing one sentence for the subject

			# get sentence number
			sentence_number = getnumber(audio_filename, 'sentence')
			
			# get ID of subject
			id_number = getnumber(audio_filename,'id')
			# print('\tSentence: %d' % sentence_number)

			# Extract formants using Praat
			df_f = extract_formants(audio_filename, sex = get_demographics(demographic, 'sex'))

			# Align phoneme using phone aligner
			df_p = align_phoneme(audio_filename, master_sentences_dict[sentence_number])

			# Tag words that were mispronounced in the file
			df_p = tag_misprounced(df_p, sentence_number, get_demographics(demographic,'mispronounced'))


			# Add id column
			df_p['id'] = id_number
			df_f['id'] = id_number

			# Add sentence number column
			df_p['sentence_number'] = sentence_number
			df_f['sentence_number'] = sentence_number

			# combine both tables together now so that we don't get duplicate columns
			df_v = voice_join(df_f, df_p)

			# This adds a column that removes X samples from the ends of each phoneme
			# we can probably play around with this number
			df_v = trim_ends(df_v, 5)

			#This extract the bandwidth
			df_v = bandwidth_extractor(audio_filename, df_v)


			# ------------------------------
			# If you have any other information that needs to be processed,
			# you should probably add it here.
			# 1) make a function that call the module that extract the information
			# 2) pass it what the whole dataframe and return the augmented dataframe
			# ------------------------------


			# Add sex column
			df_p['sex'] = get_demographics(demographic, 'sex')
			df_f['sex'] = get_demographics(demographic, 'sex')
			df_v['sex'] = get_demographics(demographic, 'sex')

			# Add accent column
			df_p['accent'] = get_demographics(demographic, 'accent')
			df_f['accent'] = get_demographics(demographic, 'accent')
			df_v['accent'] = get_demographics(demographic, 'accent')


			# Add global path of file		
			df_p['global_path'] = os.path.abspath(audio_filename)
			df_f['global_path'] = os.path.abspath(audio_filename)
			df_v['global_path'] = os.path.abspath(audio_filename)




			# Append dataframe to output handlers
			phoneme_df_frames.append(df_p)
			formant_df_frames.append(df_f)
			voice_df_frames.append(df_v)


			# --- Done processing one sentence for the subject


		# Combine All dataframes in handlers into one dataframe
		phone_df = combine_dafaframes(phoneme_df_frames) 
		formant_df = combine_dafaframes(formant_df_frames)
		voice_df = combine_dafaframes(voice_df_frames)



		#Save the results
		savedf(phone_df, 'acoustic.csv')
		savedf(formant_df, 'praat.csv')
		savedf(voice_df, 'voice.csv')


		# If it reaches here, that means no errors occurred and we can consider
		# this subject to be processed. The function below is a flag so that we
		# no longer have to process this subject next time around
		touch('_PROCESSED')
		goto_datadir() # change working directory to data directory


