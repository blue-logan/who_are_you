"""
Author: Logan Blue
Date: February 13, 2020

This file contains the main acoustic core for our vocal tract recovery work.
"""
from scipy.signal import decimate
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from transfer_function import TF
import gpu_gradient_descent as gd
import area_solver

import pdb
#pylint: disable=trailing-whitespace, invalid-name, too-many-locals
#pylint: disable=pointless-string-statement, dangerous-default-value

#GLOBAL
R_G = 1.0 #Glottis reflection coeff, assume infinite impedance
#N = 47 #Number of subpipes that make up our system
C = 354000      #Speed of sound in mm/s
#R_SERIES = [0.0] * (N-1)  + [0.714] #the reflection coeff of the system, all set to 1 to begin 
#R_KNOWN = [0.25, -0.55, 0.3, 0.25, 0.2, 0.3, 0.15, -0.1, -0.1, 0.714]
#N = len(R_KNOWN)
#R_SERIES = [round(random.randint(0, 200) / 200.0 - 1, 2) for _ in range(0, N)]
#R_END = 0.8


FS = None   #Sampling Frequency of the audio recording
FREQ_JUMP = None  #the sum between frequencies in our FFT
#TODO: we might have better luck with the average error per point (maybe with std info)
EXIT_AREA = 20    #Max area between curves allowed before we exit 

#Audio file handling
def calc_fft(data, output_size=8192, fs=44100, cutoff=None, clean=False):
    """This function will calculate the FFT of an audio waveform. Additionally it can 
    low pass filter and crop the waveform in order to provide limited coverage for 
    the transfer function that has a limited accuracy range.         
    """
    #var setup
    dec_val = 5

    #np fft
    fft = np.fft.rfft(data, output_size, norm='ortho')
    if clean:       #apply cleaning and smoothing to aid TF
        norm_factor = 2 / len(fft)
        fft = abs(fft) * norm_factor
        fft = decimate(fft, dec_val, ftype='fir', zero_phase=True)
        """Shifting to ensure that the log shift doesn't create any nan values.
        The lowest negative value will result in upwards shift by its magnitude
        plus 0.0001 ( ~ -80 dBs). """
        fft_min = fft.min()
        if fft_min < 0:
            fft = fft + abs(fft_min) + 0.0001
        #if any([x <= 0 for x in fft]):
        #    pdb.set_trace()
        fft = 20 * np.log10(fft)
        freqs = np.fft.rfftfreq(output_size, d=1.0/fs)
        freqs = decimate(freqs, dec_val, ftype='fir', zero_phase=True)

        #Smoothing the freqs (hopefully removing some noise)
        fft = gaussian_filter1d(fft, 2)

        #fft adjust to zero frequency is at zero db
        #fft = fft - fft[0] + 0.01
        #average fft around zero to make the TF more likely to find a solution
        fft = fft - np.mean(fft)

        #check lowest frequency is not negative
        if freqs[0] < 0:
            freqs[0] = 0
    else:
        fft = 20 * np.log10(fft)
        freqs = np.fft.rfftfreq(output_size, d=1.0/fs)

    if cutoff:
        max_val = int(cutoff / (fs / output_size) / dec_val)
        ans = fft[:max_val], freqs[:max_val]
    else:
        ans = fft, freqs

    return ans

def calc_ifft(data):
    """This function will calculate the inverse FFT in order to recreate a temporary waveform for 
    a given phoneme for a speakers. 
    NOTE: This function maybe replaced in future versions with a more advanced/appropriate 
    technique that allows us to more accurately mimic a speakers voice or extract additional 
    information / increase the information storage density. 
    """
    return np.fft.irfft(data, norm='ortho')

def sex_assumptions(sex):
    """
    Returns some basic, preset values that are determined by the speakers sex. 
    """
    #if sex == 'f':
    #    N = 37
    #    A_0 = 2.8
    #else:
    #    N = 44
    #    A_0 = 4.5
    #return N, A_0

    #for now we will assume a fixed value for all speakers, male and female to simplify 
    #later comparison process
    return 15, 3.7      #N, A_0

def core_main(audio, label, meta_data={}):
    """This is the main hooking point for the acoustic core. It will process a
    segment of audio data into meta data about the vocal tract or necessary
    information for later acoustic reconstruction. This function returns the
    estimated cross-sectional area series that describes the airway or the
    reconstruction meta data and the label for the information. 

    inputs: ...(audio, label, meta_data={}) 
       - audio: array - sampled audio data that is being processed
       - label: str - label for data
       - meta_data: dict - any additional information that maybe need on the other side
            of the acoustic core or additional functionality flags for later extension of 
            the core.
            Flags:
                oper: <'ext', 'gen_cc', 'gen_md'>
                    + ext - extraction mode - solve for the optimal
                            cross-sectional values
                    + gen_cc - generation mode - single pass, generation audio for a 
                            cross-sectional area series
                    + gen_md - generation mode - single pass, generated audio
                            based on meta data for now TF audio generation
                ph_type: <'vt', 'meta'> - general flags for specifying which
                            kind of extraction will be occurring. 
                    + currently being used as a stop gap for logical later in development
                FS: <int>
                    +sampling frequency for the passed in audio
                sex: <'m', 'f'> - sex of the speaker, used to determine several
                    basic assumptions throughout the process
                area_curve: np.array or list - area array for generation of audio

    outputs: acoustic_analysis data, audio out, meta_data
        - acoustic_analysis data: dict - this dataframe will contain the VT
                  cross-sectional area or the necessary information for audio
                  reconstruction that was extracted by the audio core
            + This includes the label for this current phoneme/bigram pair
            + This includes any relevant meta data that needed to be passed
                    through from the analysis
    """
    #local variables
    acoustic_data_out = None
    audio_out = None

    #number_div = 8192       #for 44100
    number_div = 5120       #for 16000


    #logical branch, which operation mode is being executed, default to ext
    operation = meta_data.get('oper', 'ext')
    if  operation == 'ext':
        #check if the extraction will be using transfer function or meta data
        if meta_data.get('ph_type', 'vt') == 'vt': 
            #for transfer function extraction
            #fft included audio data to get target for grad
            FS = meta_data.get('FS', None)
            fft_data, freqs = calc_fft(audio, output_size=number_div, \
                    fs=FS, clean=True, cutoff=5000)
 
            #initialize vars for calling gpu based gradient descent module
            N, A_0 = sex_assumptions(meta_data.get('sex', 'm'))
            R_SERIES = np.zeros((N,), dtype=np.float64)
            #R_SERIES = np.array([0.0] * N, dtype=np.float64)
    
            #call gpu GD
            #r, _, _ = gd.descent(fft_data, freqs, R_SERIES, max_iteration=1500)
            r = gd.descent(fft_data, freqs, FS, R_SERIES, max_iteration=1500)
            cc_area = area_solver.areaSolver(r, A_0)        #convert to area instead of 
                                                            #reflection coeff

            #construct outputs
            acoustic_data_out = {}
            acoustic_data_out['cross_sect_est'] = cc_area
            acoustic_data_out['label'] = label

        else:
            #for meta data
            #TODO: Solve this, probably just use low bin number FFT for now
            fft_data, freqs = calc_fft(audio, output_size=number_div, \
                    fs=meta_data.get('FS', None))

            #construct outputs
            acoustic_data_out = {}
            acoustic_data_out['fft_data'] = fft_data
            acoustic_data_out['freqs'] = freqs
            acoustic_data_out['label'] = label

    elif operation == 'gen_cc':
        #initialize vars for calling cpu based transfer function module
        area_data = meta_data.get('area_curve', None)
        r_series = area_solver.reflectionSolver(area_data)

        #call cpu TF
        """TODO: create TF object, initialize, call transfer_function
        Then create delta_t worth of audio samples to create the necessary audio"""
        #TODO: Issue in generating code creating the correct frequency response
        trans_funct = TF(r_series)
        freq_response = trans_funct.run()

        #construction output

    else: 
        #function meta_extruder to create audio from meta data
        fft_data = meta_data.get('fft_data', [])
        audio_data = calc_ifft(fft_data)
        
        #construct output

    return acoustic_data_out, audio_out
