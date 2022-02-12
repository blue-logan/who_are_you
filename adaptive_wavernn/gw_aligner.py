"""
Author: Logan Blue
Date: October 7th, 2021

This libray is being used to transcribe and force align phonemes for
audio files for the Guesswho project. There was code previously written to do
this, however, it is very poorly documented so it is being reimplemented here.

Specifically, this is being built to facilitate the creation of our adaptive
adversary tests.

Please note that due to time constraints this isn't build correctly. Really,
we should read the audio source from memory rather than a file, but since
this needs to be build quickly, I'm going to just implement this library
in the way the original code/the speech recognition library examples are
done.
"""
#pylint: disable='trailing_whitespace'

#imports
import warnings
import speech_recognition as sr
import scipy.io.wavfile as wav

#there should be a module to import, but that's not how it was done before....
import sys
sys.path.append('/home/logan/SynologyDrive/Research/guesswho_new/guesswho/code'+
        '/core/data_decorating/utils/')
from gw_utils.process import align_phoneme

#initializing objects
recogn = sr.Recognizer()

def transcribe(audio_data):
    """This function will transcribe an audio feed into text"""
    #create audiofile source, read in audio
    with sr.AudioFile(audio_data) as source:
        audio = recogn.record(source)

    #transcribe
    try:
        transcript = recogn.recognize_google(audio)
    except sr.UnknownValueError:
        #I believe this error is from the audio being too poor to be translated
        transcript = ''
        
    #return 
    return transcript

def align(audio_data, text_data):
    """This function will run out gentle based phoneme alignment tool"""
    df = align_phoneme(audio_data, text_data)

    #add additional decorations to the dataframe
    #This is original code, not cleaning
    df_wrd = df[['word_starttime','word_endtime','word']].drop_duplicates()
    df_wrd['word_starttime'] = df_wrd.apply(lambda row: int(row['word_starttime']\
            *16000),axis=1)
    df_wrd['word_endtime'] = df_wrd.apply(lambda row: int(row['word_endtime']\
            *16000),axis=1)
    df_wrd.sort_values(by=['word_starttime'])
    #df_1.to_csv(audio_file[:-4]+'.WRD', sep=' ', header=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_phn = df[['phoneme_start','phoneme_end','ipa']]
        df_phn['phoneme_start'] = df_phn.apply(lambda row:\
                int(row['phoneme_start'] * 16000),axis=1)
        df_phn['phoneme_end'] = df_phn.apply(lambda row: int(row['phoneme_end']\
                *16000),axis=1)

    return df_wrd, df_phn

def wrap_preprocess(audio_data, sr):
    """This function wraps the align and transcribe functions into a single
    call."""
    #save audio array
    wav.write('tmp_audio.wav', sr, audio_data)

    #get transcription
    transcript = transcribe('tmp_audio.wav')

    #save transcription to temp file
    with open('tmp_transcription.txt', 'w') as open_file:
        open_file.write(transcript)

    #alignment
    df_word, df_phn = align('tmp_audio.wav', 'tmp_transcription.txt')

    return df_word, df_phn
