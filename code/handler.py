"""
Author: Logan Blue
Date: February 16, 2020

This script contains the functionality and logic to operate our acoutsic
core in a number of different ways.
    - Bigram analysis - This technique is aimed at extracting more temporal information
        from a speaker speech segment.

Several inputs are needed for this function to behave as intended.
    - Labeled data: We required that the speakers phonemes in a given
    sample have already been labeled. This allows us to actually label
    the bigram.
"""
#pylint: disable=trailing-whitespace, bad-continuation, too-many-locals, invalid-name, bare-except

import os
import sys
import traceback
import numpy as np
import pandas as pd
from db_util import DBObj
from sphfile import SPHFile
from scipy.io import wavfile
import core
from tqdm import tqdm
import random
from multiprocessing import Pool

import pdb

#Global variables
WINDOW_SIZE = 565
OVERLAP = 115

def df_read_fake_csv(path):
    """Function to read in the csv containing all the necessary information for the 
    acoustic core functionalitry. 
    """
    df_read = pd.read_csv(path, sep=',',
            dtype={
                'start_word' : int,
                'end_word': int,
                'word': str,
                'sample_id': str,
                'speaker_id': str,
                'filename': str,
                'sex': str
       })
    print("Read csv done...")
    return df_read

def df_read_csv(path):
    """Function to read in the csv containing all the necessary information for the 
    acoustic core functionalitry. 
    """
    df_read = pd.read_csv(path, sep=',',
            dtype={
                'start_word' : int,
                'end_word': int,
                'word': str,
                'sample_id': str,
                'speaker_id': str,
                'start_phoneme': int,
                'end_phoneme': int,
                'arpabet': str,
                'ipa': str,
                'filename': str,
                'index_phoneme': int
       })
    print("Read csv done...")
    return df_read

def construct_audio_windows_bigram(start, div, end):
    """Set up the 5 windows of our bigram analysis
    
    TODO: decide if this is the correct way of doing this analysis
    """
    #0.50 - 0.80 of first phoneme
    first = (start + int((div - start) * 0.5), start + int((div - start) * 0.8))          
    #0.70 - 1.00 of first phoneme
    second = (start + int((div - start) * 0.7), div)                                  
    #0.80 of first - 0.20 of the second
    third = (start + int((div - start) * 0.8), div + int((end - div) * 0.2))          
    #0.00 - 0.30 of the second phoneme
    fourth = (div, div + int((end - div) * 0.3))                                      
    fifth = (div + int((end - div) * 0.2), div + int((end - div) * 0.5))              
    #0.20 - 0.50 of the second phoneme
    return first, second, third, fourth, fifth

def construct_uniform_windows_ph(start, div, end):
    """Set up the windows for our windows around the center divide for each 
    bigram. This function obviously requires knownledge of the individually 
    labeld phonemes.
    """
    windows = []
    #forward of divide
    curr_front = div - int(WINDOW_SIZE / 2.0)
    while start < curr_front:
        windows = [(curr_front, curr_front + WINDOW_SIZE)] + windows    #prepending
        curr_front -= WINDOW_SIZE - OVERLAP

    #behind divide
    curr_front = div - int(WINDOW_SIZE / 2.0)
    curr_front += WINDOW_SIZE - OVERLAP
    while end - WINDOW_SIZE > curr_front:
        windows = windows + [(curr_front, curr_front + WINDOW_SIZE)]    #appending
        curr_front += WINDOW_SIZE - OVERLAP
    return windows

def construct_uniform_windows_words(start, end):
    """Set up windows for our windows linearly through the space. This function assumes that 
    the individual phoneme timings are unknown and that only the word timings are avaible. 
    """
    windows = []
    curr_front = start
    while curr_front + WINDOW_SIZE < end:
        windows.append((curr_front, curr_front + WINDOW_SIZE))
        curr_front = curr_front + (WINDOW_SIZE - OVERLAP)

    return windows

def bigram_analysis_word(inputs):
    """This function wraps the acoustic core to in order to perform a
    bigram analysis of provided audio. This technique looks at the temporal
    behavior as a speaker moves from one phoneme to the next. Ideally this should
    allow us to pull out a larger amount of the variation and the unique aspects 
    of how a individual articulates their speech.

    This version of this method is for running the analysis when the individual 
    phoneme timings are not known. Thus we will linearly run this analysis over each
    individual word.

    Inputs (this come in as an individual tuple that needs to be broken out in order to be
    processed. The tuple input in needs for compatibility with python Pools)
        df_audio: dataframe that contains the labeled phonemes
        db: the object that allows access to the mongodb back end
        path: the filepath this function is currently responsible for processing


    NOTE: This function is specifically for the deep faked audio data. Instead of looping 
    through words and bigrams, we will instead just using the sliding window over all 
    pre-labeled words in the dataset.
    """
    #extract parameters
    df_audio, path, data_name = inputs

    #load the db to enter values
    #db = DBObj(collection_name='var_test')      #var_test
    #db = DBObj(collection_name='fakes_test')      #deep faked audio
    db = DBObj(collection_name=data_name)       #main data run
    
    #process one audio file at a time
    #load audio file, timit data is actually sphere files...
    #but our deepfaked audio is actually a wav file....
    fs, curr_audio = wavfile.read(path)
        
    #loop through all records related to current audio file
    grouped_df = df_audio[df_audio.filepath == path].groupby(['speaker_id', 'sample_id'])
    for key, df_cword in grouped_df:
        #sliding window over words to analyze deep fakes more easily
        for word in df_cword.word.unique():
            c_word = df_cword[df_cword.word == word]
            #get analysis windows
            """Original 5 window process
            windows = construct_audio_windows_bigram(first.start_word.values[0], \
                    second.end_word.values[0])
            """
            #get constant sliding window
            windows = construct_uniform_windows_words(c_word.start_word.values[0], \
                    c_word.end_word.values[0])

            #construct meta info for the acoustic core before we run files
            core_meta = {'oper':'ext', 'ph_type':'vt', 'FS':fs, 'sex':c_word.sex.values[0]}
            #for each window, run the Acoustic core for each segment
            for win_index in range(0, len(windows)):
                win = windows[win_index]
                try:
                    acoustic_data, _ = core.core_main(curr_audio[win[0]: win[1]], '--', core_meta)
                                    #add other relevant data to acoustic_data dictionary
                    acoustic_data['filepath'] = path
                    acoustic_data['speaker_id'] = df_cword['speaker_id'].values[0]
                    acoustic_data['window_start'] = int(win[0])
                    acoustic_data['window_end'] = int(win[1])
                    acoustic_data['window_index'] = win_index
                    acoustic_data['sex'] = c_word.sex.values[0]
                    #sys.exit()
                    #write data into db base one at a time
                    db.insert(acoustic_data)     
                except:
                    #print("windows: ", win[0], " --- ", win[1], ": size: ", win[1] - win[0])
                    #pd.set_option('display.max_columns', None)
                    #pd.set_option('display.max_colwidth', -1)
                    #pd.set_option('display.max_rows', 20)
                    #print("group: ", df_cword)
                    #print("first: ", first)
                    #print("second: ", second)
                    #print("path: ", path)
                    #print("Label: ", label) 
                    traceback.print_exc()
                    sys.exit()

def bigram_analysis_phoneme(inputs):
    """This function wraps the acoustic core to in order to perform a
    bigram analysis of provided audio. This technique looks at the temporal
    behavior as a speaker moves from one phoneme to the next. Ideally this should
    allow us to pull out a larger amount of the variation and the unique aspects 
    of how a individual articulates their speech.

    This version of this method is for running the analysis when the individual 
    phoneme timings are known. 

    Inputs (this come in as an individual tuple that needs to be broken out in order to be
    processed. The tuple input in needs for compatibility with python Pools)
        df_audio: dataframe that contains the labeled phonemes
        db: the object that allows access to the mongodb back end
        path: the filepath this function is currently responsible for processing
    """
    #extract parameters
    df_audio, path, data_name = inputs

    #load the db to enter values
    #db = DBObj(collection_name='var_test')      #var_test
    #db = DBObj(collection_name='fakes_test')      #deep faked audio
    db = DBObj(collection_name=data_name)       #main data run
   
    #process one audio file at a time
    #load audio file, timit data is actually sphere files...
    sph = SPHFile(path)
    curr_audio = sph.content
    fs = sph.format['sample_rate']
        
    #loop through all records related to current audio file
    grouped_df = df_audio[df_audio.filepath == path].groupby(['speaker_id', 'sample_id',\
            'word'])
    for key, _ in grouped_df:
        df_cword = grouped_df.get_group(key)
        #sliding window over phoneme index values to pull out bigram pairs
        #do not use last index since it will not be related to the phoneme (should 
        #be a space between them)
        for index in range(0, df_cword.index_phoneme.max()):
            #get bigram window --> start, phoneme_div, end
            first = df_cword[df_cword.index_phoneme == index]
            second = df_cword[df_cword.index_phoneme == index + 1]
            #This prevent breaks from epenthesis silence from passing values
            #off as bigrams (the silence interrupts them bigram, so we may not
            #get a good/representative transition)
            try:
                label = first.ipa.values[0] + " -- " + second.ipa.values[0]

                #get analysis windows
                """original 5 window bigram division
                windows = construct_audio_windows_bigram(first.start_phoneme.values[0], \
                        first.end_phoneme.values[0], second.end_phoneme.values[0])
                """
                windows = construct_uniform_windows_ph(first.start_phoneme.values[0], \
                        first.end_phoneme.values[0], second.end_phoneme.values[0])
                
                #construct meta info for the acoustic core before we run files
                core_meta = {'oper':'ext', 'ph_type':'vt', 'FS':fs, 'sex':first.sex.values[0]}
                #for each window, run the Acoustic core for each segment
                for win_index in range(0, len(windows)):
                    win = windows[win_index]
                    try:
                        acoustic_data, _ = core.core_main(curr_audio[win[0]: win[1]], 
                                label, core_meta)
                        #add other relevant data to acoustic_data dictionary
                        acoustic_data['filepath'] = path
                        acoustic_data['speaker_id'] = df_cword['speaker_id'].values[0]
                        acoustic_data['start_bigram'] = int(df_cword['start_phoneme'].values[0])
                        acoustic_data['end_bigram'] = int(df_cword['end_phoneme'].values[0])
                        acoustic_data['window_start'] = int(win[0])
                        acoustic_data['window_end'] = int(win[1])
                        acoustic_data['window_index'] = win_index
                        acoustic_data['sex'] = first.sex.values[0]
                        #sys.exit()
                        #write data into db base one at a time
                        db.insert(acoustic_data)     
                    except:
                        #print("windows: ", win[0], " --- ", win[1], ": size: ", win[1] - win[0])
                        #pd.set_option('display.max_columns', None)
                        #pd.set_option('display.max_colwidth', -1)
                        #pd.set_option('display.max_rows', 20)
                        #print("group: ", df_cword)
                        #print("first: ", first)
                        #print("second: ", second)
                        #print("path: ", path)
                        #print("Label: ", label) 
                        traceback.print_exc()
                        sys.exit()
            except:
                pass            #just ingore this and move on
 
def bigram_multi(df_audio, data_name=None):
    """This function wraps the acoustic core to in order to perform a
    bigram analysis of provided audio. This technique looks at the temporal
    behavior as a speaker moves from one phoneme to the next. Ideally this should
    allow us to pull out a larger amount of the variation and the unique aspects 
    of how a individual articulates their speech.
    
    Inputs:
        df_audio : dataframe that contains the labeled phonemes, file paths to relevant 
            audio files, and timing information for a given speakers speech sample. 
    """
    #filtering for smaller dataset for certain tests
    #print("Filtered incoming data for testing...")
    #filter out already processed speakers for adding to our processed speaker db
    #df_audio = df_audio[~df_audio.speaker_id.isin(['MJEB0', 'MGSH0', 'MGAK0', 'MVRW0'])]
    
    #select only already processed speakers for the variability tests
    """=====This is removed for non-timit based datasets===="""
    #df_audio = df_audio[df_audio.speaker_id.isin(['MJEB0', 'MGSH0', 'MGAK0',
    #'MVRW0', 'FDNC0', 'FSMA0', 'MRGS0', 'MRWS0', 'FMKF0', 'MKAH0', 'FMJB0',
    #'FCYL0', 'MRJH0', 'MKLS0', 'MMDS0', 'FALK0', 'FSKC0', 'MDLH0', 'MGAF0',
    #'MDBB1', 'MLNS0', 'FDJH0', 'MRJB1', 'MHJB0', 'MDNS0', 'MDEF0', 'FCAG0',
    #'FJWB1', 'MJLS0', 'MTRT0', 'FGDP0', 'FEXM0', 'MDSJ0', 'MJRG0', 'MTRC0',
    #'MJEE0', 'MMVP0', 'MJDM0', 'MPMB0', 'MTDP0', 'MSDS0', 'FBCH0', 'MBOM0',
    #'MRML0', 'MRMG0', 'FCRZ0', 'MMPM0', 'MDLR1', 'FJDM2'])]


    #total_speakers = len(df_audio.speaker_id.unique())
    #set sample seed to ensure we get the same selections for the variability test
    #random.seed(1012)
    #subset_samples = random.sample(list(df_audio.speaker_id.unique()), 10)
    #print("ID's selected: ", subset_samples)
    #df_audio = df_audio[df_audio.speaker_id.isin(subset_samples)] 

    #prep data_base access info
    if not data_name:
        data_name = input("What collection do the results need to be in: ")
    else:
        print("Data will be stored in the ", data_name, " collection. Is this OK? [Y/n]")
        ans = input()
        if ans == 'n':
            sys.exit()
        else:
            print("Proceeding...")

    while True:
        ans = input("Does the metadata for this audioset contained prelabeled " + \
                "phoneme positions? [y/n]: ")
        if ans.lower() == 'y':
            labeled_phoneme = True
            break
        elif ans.lower() == 'n':
            labeled_phoneme = False
            break
        else:
            print("Invalid input, please answer the question\n")

    #create input list
    inputs = []
    for path in df_audio.filepath.unique():
        inputs.append((df_audio, path, data_name))

    #multithreading process
    print("Number of files to process: ", len(inputs))
    print("Pool starting...")
    #with Pool(5) as p:
    #    if labeled_phoneme:
    #        p.map(bigram_analysis_phoneme, inputs)
    #    else:
    #        p.map(bigram_analysis_word, inputs)
    for tmp in inputs:
        bigram_analysis_phoneme(tmp)
               
def main():
    """Function that allows the handler to act as a terminal interface. 
    input: 
        -function:
            - 'bi'

        -csv to load that contains the phoneme labels, audio file locations, and 
        additional data necessary for indicated function flag
    """
    #extract args and function
    try:
        funct = sys.argv[1]
        if funct == 'bigram':
            if len(sys.argv) > 3:
                data_name = sys.argv[2]
                path_df = sys.argv[3]
                #hook to function & check if this is a fake dataset or 
                while True:
                    ans = input("Is this a dataset of deepfakes? [y/n]: ")
                    if ans.lower() == 'y':
                        bigram_multi(df_read_fake_csv(path_df), data_name)
                        break
                    elif ans.lower() == 'n':
                        bigram_multi(df_read_csv(path_df), data_name)
                        break
                    else:
                        print("Invalid response, please answer the question.\n")
            else:
                #hook to function
                path_df = sys.argv[2]
                bigram_multi(df_read_csv(path_df))
        else:
            print("Unrecognized command")
    except:
        #print("Windows local directory: ", os.getcwd())
        print("Error: ")
        traceback.print_exc()

if __name__ == "__main__":
    main()
