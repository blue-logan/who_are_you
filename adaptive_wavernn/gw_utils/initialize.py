from __future__ import print_function
from random import shuffle
import os,sys
import glob,json

if len(sys.argv) == 1:
	# print('please put directory name')
	path = None
else:
	path = '/'.join(['actual_data',sys.argv[1]])


starters = [
	'Hey Siri, how is the weather?',
	'Alexa, what is two plus two'
]

def getNextID():
	# example of item in id_paths --> actual_data/person/id_6.dat
	id_paths = glob.glob('actual_data/*/*.dat') #get all id paths
	id_files = [path.split('id_')[-1] for path in id_paths] #get id string
	ids = [int(name.replace('.dat','')) for name in id_files] # clean to get id for each filename
	# print('Max ID: %d' % max(ids))
	return max(ids) + 1 #return the next available id

def touch(fname):
	# same as touch filename in command line
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


# print(getNextID())
# sys.exit()

def printlist(x):
	for y in x:
		print(y)
		print()


def make_demographics_template():
	demographics = {
		'sex'    : '',
		'accent' : '',
		'mispronounced': {
			"":[""],
		}
	}
	with open('demographic.json', 'w') as fn:
		json.dump(demographics, fn, indent = 2)


#Get next available ID
subject_id = getNextID()


# Shuffle sentences
sentences = [ line.strip() for line in open('master_sentences.txt','r').readlines()]
cleaned_sentence = [line.split(')')[1][1:] for line in sentences]
shuffle(cleaned_sentence)
to_print = starters + cleaned_sentence

# printlist(to_print)

if path is None:
	#if no specific name was entered, then make the name be person[ID]
	name = 'person' + str(subject_id)
	path = '/'.join(['actual_data',name])
	print('No specific name chosen - saving in diretory %s' % path)

#Make subject directory and change to working directory
os.mkdir(path)
os.chdir(path)


#make ID file
touch('id_%d.dat' % subject_id)

#make demographic file
make_demographics_template()

print('Subject ID: %d' % subject_id)


#write shuffled sentences
with open('sentences.txt', 'w') as f:
	for line in to_print:
		f.write(line +'\n\n')


print('Ready to start recording')
