from sphfile import SPHFile
sph =SPHFile('../../../Data_Stores/Timit/Data/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV')
# Note that the following loads the whole file into ram
print( sph.format )
# write out a wav file with content from 111.29 to 123.57 seconds
sph.write_wav( 'timit_test.wav', 111.29, 123.57 )
