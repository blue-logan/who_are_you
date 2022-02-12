#These need to be imported so that matplotlib doesnt scream about
#not finding a display window appear (e.g., when it run in the server)
import matplotlib
matplotlib.use('Agg')

import pdb
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Additional functions needed for the bandwidth extractor"""
#find nearest time value to our start and end times
def find_nearest(times, start, end=None):
    #we look for both start and end at the same time to prevent having to walk the list twice
    start_index = -1
    start_min_err = -1
    end_index = -1
    end_min_err = -1
    #check if the search is for one or two values
    if end != None:
        #loop through times until we start to move away from both end and start
        for i in range(0,len(times)):
            #start
            curr_err = abs(start - times[i])
            if start_min_err < 0:
                start_min_err = curr_err
                start_index = i
            elif curr_err < start_min_err:
                start_min_err = curr_err
                start_index = i
            #else:        #once we start missing a setting attempt, we have passed it and its time to leave
            #    break
            #end
            curr_err = abs(end - times[i])
            if end_min_err < 0:
                end_min_err = curr_err
                start_index = i
            elif curr_err < end_min_err:
                end_min_err = curr_err
                end_index = i
            #else:        #once we start missing a setting attempt, we have passed it and its time to leave
            #    break
        return start_index, end_index
    else:
        #loop through times until we start to move away from both end and start
        for i in range(0,len(times)):
            #start
            curr_err = abs(start - times[i])
            if start_min_err < 0:
                start_min_err = curr_err
                start_index = i
            elif curr_err < start_min_err:
                start_min_err = curr_err
                start_index = i
            #else:        #once we start missing a setting attempt, we have passed it and its time to leave
            #    break
        return start_index
    

#find the first place the curve is below the curve value
#TODO: Test this code
def first_below(search_space, cutoff, max_index, lower=True):
    if lower:
        #we will look for the lower bandwidth edge
        for i in range(max_index-1, -1, -1):
            #check if the search space value is beneath the cutoff value
            if search_space[i] < cutoff:
                #once it is below, return the index value
                return i
        #we have not found a vaue beneath this point, this means it is mostly that 
        #this point is near one of the edges
    else:
        #we will look for the upper bandwidth edge
        for i in range(max_index+1, len(search_space)):
            #check if the search space value is beneath the cutoff value
            if search_space[i] < cutoff:
                #once it is below, return the index value
                return i
        #we have not found a vaue beneath this point, this means it is mostly that 
        #this point is near one of the edges
                
#slope calculator
def slope(x_l, y_l, x_h, y_h):
    return (y_h - y_l) / (x_h - x_l)

#This function finds a the maximum near this index of this array
def find_max_near(data, index):
    #use slope to walk along the array
    #calc slope for the point to the left, if negative, head that way (left)
    #we have 4 case, already max (+, -) [read positive slope to the left, negative to the right]
    #            local min (-, +), down slope (-, -), and up slope (+, +)
    left_slope = slope(index-1, data[index-1], index, data[index])
    right_slope = slope(index, data[index], index+1, data[index+1])
    
    if left_slope > 0 and right_slope < 0:    #local max, do not move
        return index 
    elif left_slope > 0 and right_slope > 0:      #up slope, move right
        #iterate that direction until the slope's direction changes
        index += 1
        curr_slope = slope(index, data[index], index+1, data[index+1])
        while curr_slope > 0:
            index+=1
            curr_slope = slope(index, data[index], index+1, data[index+1])
        return index
    
    #if either on a down slope or at a local min, move to the left
    else: 
        #iterate that direction until the slope's direction changes
        index -= 1
        curr_slope = slope(index-1, data[index-1], index, data[index])
        while curr_slope < 0:
            index-=1
            curr_slope = slope(index-1, data[index-1], index, data[index])
        return index


def load_audio_file(filepath):
	from scipy.io.wavfile import read
	return read(filepath)

"""Bandwidth extractor"""
def bandwidth_extractor(file_path, df_data):
    #load date for ripping the whole file
    sr, samples = load_audio_file(file_path)

    #create spectrogram of the whole file
    spec, freq, t, im = plt.specgram(samples, NFFT=256, cmap='Blues', Fs=sr, noverlap=128)     #, scale='dB')
    new_spec = spec.transpose()      #flip the spectrogram for easy access to times
    
    #now loop through all the found formants in the df_data for this file
    rows = []
    for index, row in df_data.iterrows():
        #find the relevant rows to my phoneme
        start = find_nearest(t, row['time'])
        ph = new_spec[start]

        #re-flip for easy access to frequencies
        ph = ph.transpose()

        #sum the time components together
        #ph_b = [sum(ph_b_i) for ph_b_i in ph_b]
        ph = ph.tolist()

        #find the bandwidth value for each of the formants
        bandwidth_list = []
        bandwidth_list.append(row['time'])
        for fn in ['f1','f2','f3','f4','f5']:
            form_index = find_max_near(ph, find_nearest(freq, row[fn]))

            highest = ph[form_index]
            width_point = 0.7079 * highest
            #get the samples that are closest to the bandwidth value (always lower)
            band_index_low = first_below(ph, width_point, form_index)
            band_index_hi = first_below(ph, width_point, form_index, lower=False)
            
            #check that both first belows returned a value, if one didnot, it is because
            #the list did not go far enough
            if band_index_low != None:
                #find slope between max and the lower bound
                rise = highest - ph[band_index_low]
                run = freq[form_index] - freq[band_index_low]
                low_slope = rise / run

                #find the bandwidth frequency value
                band_lower = freq[form_index]-(highest - width_point) / low_slope
            if band_index_hi != None:
                #reapeat for high side
                rise = ph[band_index_hi] - highest 
                run = freq[band_index_hi] - freq[form_index]
                high_slope = rise / run
                band_high = (width_point - highest) / high_slope + freq[form_index]
            
            #calc bandwidth even if one side failed to fine purchase
            if band_index_hi == None:
                #calculate band width values
                bandwidth_list.append(2*(row[fn]-band_lower))
            elif band_index_low == None:
                #calculate band width values
                bandwidth_list.append(2*(band_high-row[fn]))
            else:
                #calculate band width values
                bandwidth_list.append(band_high-band_lower)
        #turn bandwidths into a df, include the time from the original df to merge on later
        rows.append(bandwidth_list)
        
    #merge the df_data and the new df_bands into a single dataframe and return it
    rows = np.asarray(rows).transpose()
    df_bands = pd.DataFrame({'time':rows[0], 'bandwidth1':rows[1], 'bandwidth2':rows[2], \
                             'bandwidth3':rows[3], 'bandwidth4':rows[4], 'bandwidth5':rows[5]})
    return df_data.merge(df_bands, on='time')
