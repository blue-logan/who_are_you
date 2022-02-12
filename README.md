# who_are_you
This repository contains the code associated with the paper: "Who Are You (I Really Wanna Know)? Detecting Audio DeepFakes Through Vocal Tract Reconstruction" that will be presented at USENIX Secuirty 2022. 

## Abstract 
Generative machine learning models have made convincing voice synthesis a
reality. While such tools can be extremely useful in applications where people
consent to their voices being cloned (e.g., patients losing the ability to
speak, actors not wanting to have to redo dialog, etc), they also allow for the
creation of nonconsensual content known as deepfakes. This malicious audio is
problematic not only because it can convincingly be used to impersonate
arbitrary users, but because detecting deepfakes is challenging and generally
requires knowledge of the specific deepfake generator. In this paper, we develop
a new mechanism for detecting audio deepfakes using techniques from the field of
articulatory phonetics.  Specifically, we apply fluid dynamics to estimate the
arrangement of the human vocal tract during speech generation and show that
deepfakes often model impossible or highly-unlikely anatomical arrangements.
When parameterized to achieve 99.9% precision, our detection
mechanism achieves a recall of 99.5%, correctly identifying all but
one deepfake sample in our dataset. We then discuss the limitations of this
approach, and how deepfake models fail to reproduce all aspects of speech
equally. In so doing, we demonstrate that subtle, but biologically constrained
aspects of how humans generate speech are not captured by current models, and
can therefore act as a powerful tool to detect audio deepfakes.

## Detector Operation
Running the main vocal tract estimator is done using the handler.py script in the core directory. 
  -- python handler bigram <test_name> <data_location>
      - bigram: the mode the code base will be running in. 
      - test_name: the name of the current testing being run. The detector will attempt to store the results of its operations in a mongo Data Base running locally. By default it will look for a DB names exploration and will create a new collection called test_name.
      - data_location: the local path to a CSV containing the meta data needed for processing.
   
## Current Detector Limitations
Currently, this code leverages Nvidia Cuda libraries running specifically on a GTX 1080. No additionally GPU compatibility has been tried or tested at this time. In the near future we will be adding additions to this repo to allow anyone to run this code, however, before such time it will likely be difficult to this detector function on a machine without a 1080. 
      
## Metadata Generation
The detector needs metadata csv file to work. This file describes the phonetic divisions within audio files and will allow the detector to work on only small pieces of audio at at time. If you would like to create your own, they will need the following columns. 
  - start_word: sample index in the audio of the beginning of the word   
  - end_word: sample index in the audio of the end of the word 
  - word: word the bigram belongs to
  - sample_id: unique id of the sample
  - speaker_id: unique id of the speaker
  - start_phoneme: starting phoneme in the bigram
  - end_phoneme: ending phoneme in the bigram
  - sex: sex of the speaker
  - arpabet: arpabet label of the bigram
  - ipa: IPA label fo the bigram
  - filepath: local location of the audio sample the bigram is in
  - index_phoneme: index of bigram within the word

Ex: 
 ,start_word,   end_word,   word,       sample_id,      speaker_id, start_phoneme,  end_phoneme,    sex,    arpabet,    ipa,    filepath,                   index_phoneme
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    50880,          53120,          f,      s,          s,      ../data/LA_E_2039915.wav,       0
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    53120,          54240,          f,      k,          k,      ../data/LA_E_2039915.wav,       1
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    54240,          55360,          f,      aa,         ɑ,      ../data/LA_E_2039915.wav,       2
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    55360,          56320,          f,      t,          t,      ../data/LA_E_2039915.wav,       3
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    56320,          56960,          f,      l,          l,      ../data/LA_E_2039915.wav,       4
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    56960,          58080,          f,      ah,         ʌ,      ../data/LA_E_2039915.wav,       5
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    58080,          59519,          f,      n,          n,      ../data/LA_E_2039915.wav,       6
 ,50880,        60000,      scotland,   LA_E_2039915,   ch_2/LA,    59519,          59999,          f,      d,          d,      ../data/LA_E_2039915.wav,       7

We will be adding code to automatically generating these metadata files and much needed documentation in the near future.
