<!-- written by Luis -->
This is the prepocessing pipeline for the data.

Everything is actually being run in Kramer but it has symlinks
to a copy of the repo I have in my home directory (in Kramer).

Kramer data path: 
	`/home/lvargas/shared/audio/guesswho` (this has symlinks to repo) 
Kramer repo path:
	`/home/lvargas/researchProjects/guesswho18`

Probably a TODO is to change those symlinks to something owned by Logan


The structure of the preprocess (`/guesswho18/code/preprocess`) pipe is as follows:
	- `initialize.py`
		This was used during our data collection process. It creates a new person directory and gives them id, sentence order, etc...
		You probably will not use this unless we collect more data
	- `build_master.py`
		This is used to concat individual person files into one master csv. If you run `process.py` this should be ran to update the master csv files
	- `config.py`
		contains paths of root directory, data directory, ipa-arpa converter dict
	- `process.py`
		- This is the main file that will process each person to get phonemes, formants, bandwidth, etc.. if you have more information that you need to extract (that is not analysis), you should probably add a call here. To see how it works open the file because it is very well documented. 
		- *An important note is that for this to run, the phonemizer services needs to be running as well. This can be found on Kramer `/home/lvargas/shared/audio/guesswho/gentle` and runs by using `python3 serve.py`*
		- Usually if I want to add more functionality to this process I put the functionality into its own file (e.g., bandwidth extractor, phoenemizer, praat) in the `preprocess` sub directory and only make a function call in this file. You'll see what I mean when you open and read the comments.
	- `preprocess/bandwith_extractor.py`
		- This script extracts bandwidths and is used in `process.py`
	- `preprocess/demographic.py`
		- empty file: need to be removed. Functionality was transfered elsewhere
	- `preprocess/phonemizer.py`
		- This script takes care of extracting the phonemes and is used in `process.py`, for this to work, you need to have the `gentle` service running.
	- `preprocess/praatformant.py`
		- This script is a python wrapper that called praat functionality and is used in `process.py`
	- `preprocess/get_formants.praat`
		- This is the praat script we used to get the formants. The only modification form the original one is the ability to extract f4 and f5
		- Input changes to this script should be made on the python wrapper.
