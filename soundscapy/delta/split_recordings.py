#%%
# use pydub to split the wav and mp3 recordings approximately in half
# TODO: Need to work out the precise rules we want to use
# NOTE: My thoughts are to aim for 13s length. Procedure is to read in the full recording,
# Read in the wav file, calculate the number of full 13s samples that could be extracted, 
# pull those out as CT101-1.wav, CT101-2.wav, etc. and discard the rest of the recording.

from pydub import AudioSegment
test_file = "/media/sf_R_DRIVE/UCL_SSID/DeLTA/data/mp3_exports/NonLockdown/CT113.mp3"
rec = AudioSegment.from_mp3(test_file)
rec_cut = rec[:13*1000]




# %%
