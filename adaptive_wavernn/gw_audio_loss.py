"""
Author: Logan Blue
Date: October 7, 2021

This function's purpose is to prep the audio for the guesswho backend to process
during the adaptive adversary tests.
"""
#pylint: disable='trailing-whitespace'
import sys
import pdb
import traceback
import pandas as pd
import torch

import gw_core
from gw_aligner import wrap_preprocess

#Global vars
WINDOW_SIZE = 565
OVERLAP = 115
SR = 16000


#============== copied form csv_master_creator in code/core ==================
# arpabet to ipa conversion
arpa_key_master = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'eh', 'er',
        'ey', 'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux' , 'b', 'ch', 'd',
        'dh', 'dx', 'el', 'em', 'en', 'f', 'g', 'h', 'hh', 'jh', 'k', 'l', 'm',
        'n', 'ng', 'nx', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'wh',
        'y', 'z', 'zh', 'ax-h', 'bcl', 'dcl', 'eng', 'gcl', 'hv', 'kcl', 'pcl',
        'tcl', 'pau', 'epi', 'h#']
ipa_key_master = ['ɑ', 'æ', 'ʌ', 'ɔ', 'aʊ', 'ə', 'ɚ', 'aɪ', 'ɛ', 'ɝ', 'eɪ',
        'ɪ', 'ɨ', 'i', 'oʊ', 'ɔɪ', 'ʊ', 'u', 'ʉ', 'b', 'tʃ', 'd', 'ð', 'ɾ',
        'l̩', 'm̩', 'n̩', 'f', 'ɡ', 'h', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'ŋ', 'ɾ̃',
        'p', 'ʔ', 'ɹ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'ʍ', 'j', 'z', 'ʒ', 'ə̥',
        'b̚', 'd̚', 'ŋ̍', 'ɡ̚', 'ɦ', 'k̚', 'p̚', 't̚', 'N/A', 'N/A', 'N/A']

ipa_conversion = dict(zip(arpa_key_master, ipa_key_master))

def convert_to_ipa(arpa_key):
    """Convert arphabert to ipa characters"""
    output = []
    for key in arpa_key:
        output.append(ipa_conversion[key])
    return output

arpa_conversion = dict(zip(ipa_key_master, arpa_key_master))
def convert_from_ipa(ipa_key):
    """Convert ipa to arphabet characters"""
    output = []
    for key in ipa_key:
        output.append(arpa_conversion.get(key, 'N/A'))
    return output

def join_word_phoneme(df_wrd, df_phn, audio_file):
    """Join my word and phoneme dataframes"""
    new_df = df_wrd.merge(df_phn, 'outer', on=('sample_id', 'speaker_id'), \
                          suffixes=('_word', '_phoneme'))
    new_df = new_df[(new_df.start_phoneme >= new_df.start_word) &
                    (new_df.end_phoneme <= new_df.end_word)]
    try:
        new_df['ipa'] = convert_to_ipa(new_df['arpabet'])
        new_df['filepath'] = audio_file
    except:
        pdb.set_trace()
    if audio_file[audio_file.rfind('/') - 5] == 'M':
        new_df['sex'] = 'm'
    else:
        new_df['sex'] = 'f'
    return new_df

def ipa_join_word_phoneme(df_wrd, df_phn):
    """add new columns that we will be merging over"""
    df_phn['start_word'] = -1
    df_phn['end_word'] = -1
    df_phn['word'] = 'NONE'

    #walk df_phn, updating each row accordingly
    for index, row in df_phn.iterrows():
        wrd_row = df_wrd[(df_wrd.word_starttime <= row.phoneme_start) &\
                (df_wrd.word_endtime >= row.phoneme_end)].reset_index(drop=True)
        try:
            df_phn.loc[index, 'start_word'] = wrd_row.loc[0, 'word_starttime']
            df_phn.loc[index, 'end_word'] = wrd_row.loc[0, 'word_endtime']
            df_phn.loc[index, 'word'] = wrd_row.loc[0, 'word']
        except:
            #failed to match a phone to a word
            pass

    #add in everything else the function might need
    df_phn['arpabet'] = convert_from_ipa(df_phn['ipa'])
    df_phn['filepath'] = 'N/A'
    df_phn['sample_id'] = 'model_loss'
    df_phn['speaker_id'] = 'model_loss'

    #add index_phoneme
    df_phn['index_phoneme'] = -1
    group_df = df_phn.groupby('start_word')
    for _, item in group_df:
        new_indices = list(range(len(item.index_phoneme)))
        df_phn.loc[item.index, 'index_phoneme'] = new_indices

    #rename phoneme columnds
    df_phn = df_phn.rename(columns={'phoneme_start':'start_phoneme', 
                                    'phoneme_end': 'end_phoneme'})

    #clean up df of things that didn't line up
    df_phn.drop(df_phn[df_phn.end_word == -1].index, inplace=True)

    return df_phn

#=============================================================================

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

def get_estimate(audio_data, fs):
    """This function is going to wrap my core functions of GW to get the
    vocal tract estimation for all phonemes in a sample."""
    #transcribe and align
    df_wrd, df_phn = wrap_preprocess(audio_data, fs)

    #group together bigrams
    df_meta = ipa_join_word_phoneme(df_wrd, df_phn)

    #now that I have the meta data and the audio file, we can start to segment
    #it into window and pass it to the backend core
    #groupby word
    grouped_df = df_meta.groupby('word')

    output = []
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
            #off as bigrams (the silence interrupts the bigram, so we may not   
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
                core_meta = {'oper':'ext', 'ph_type':'vt', 'FS':SR, 
                        'sex':'f'}

                #for each window, run the Acoustic core for each segment         
                for win_index in range(0, len(windows)):                         
                    win = windows[win_index]
                    try:
                        acoustic_data, _ = core.core_main(audio_data[win[0]:\
                                win[1]], label, core_meta)
                        output.append(acoustic_data)
                    except:
                        print('Broken at core')
                        traceback.print_exc()
                        sys.exit()
            except:
                print('Broken in windowing and bigram creation')
                pass

    return pd.DataFrame(output)

def calc_loss(organic_audio, synthetic_audio, fs):
    """This function will by my main hook into calculating the loss. Since I'm
    not sure if the batching is going to provide me matched audio (i.e., a 
    synthetic sample of a phrase and the organic equivalent) or not, we are going
    to assume that the two _audio samples are lists of the audio samples from
    the batch. There for, we are going to use the organic_audio as the goal and
    then calculate the distance the synthetic_audio is from it."""
    #process organic_audio samples
    df_org = []
    for sample in organic_audio:
        df_org.append(get_estimate(sample, fs))

    #process synthetic_audio sample
    df_syn = []
    for sample in synthetic_audio:
        df_syn.appned(get_estimate(sample, fs))

    #move to full dfs
    df_org = pd.concat(df_org, ignore_index=True)
    df_syn = pd.concat(df_syn, ignore_index=True)

    #check my organic audio for repeat labels
    org_aves = []
    for key, grp in df_org.groupby('label'): 
        #each repeat label, average and create a master value
        #get values
        vals = np.array(list(grp.cross_sect_est.values))

        #get averages
        aves = np.mean(vals, axis=0)

        #save
        org_aves.append({'label':key, 'cross_sect_est':aves})


    #for synthetic samples
    err = []
    for _, row in df_syn.iterrows():
        #find distance from the organic average and summ across samples
        err.append(abs(row.cross_sect_est - \
                org_ave[org_ave.label == row.label].cross_sect_est.values[0]\
                / len(row.cross_sect_est))

    #return the average error per tube per bigram as our err
    return torch.mean(err)
