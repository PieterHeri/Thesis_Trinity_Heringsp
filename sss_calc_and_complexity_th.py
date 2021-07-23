# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:55:52 2021

Determining average swing, stance, stride and complexity index from Shimmer 3 IMU data 

Pieter Herings
Trinity College Dublin, University of Dublin

"""
#from pyEntropy import multiscale_entropy
from scipy import signal
import scipy as sc
import numpy as np
from scipy.io import loadmat
from statistics import stdev
import matplotlib.pyplot as plt
from numpy import trapz

def MATLAB_to_nparray(patient):
    """
    Input: patient = string of filename as .MAT (IMU data)
    Output: output_array = np.ndarray (IMU data)
    """
    data_pt = loadmat(patient)                                                 #load MATLAB file as data_pt (dictionary)                                                                
    x = str(patient[:6])                                                       #get rid of .mat, store as x
    output_array = data_pt[x]                                                  #select only sensordata from dictionary
    
    return output_array

#------------------------------------------------------------------------------
def get_data_patient(mat_pt, shimmer_column):
    """
    Input:
    mat_pt = string of filename patient (.mat)
    shimmer_volumn = refers to which output of device you want to analyse;
    column 0: Timestamp
    column 1-3: x, y, z low noise accelerometer
    column 4-6: x, y, z wide range accelerometer
    column 7-9: x, y, z gyroscope
    
    Output:
    timestamps = numpy.ndarray of all timestamps, starting at 0 in milliseconds
    sp_shimmer_output = numpy.ndarray of a specific sensor output
    hertz = numpy.float64
    """

    total_data = MATLAB_to_nparray(mat_pt)                                     #use function (line 44), loads matlab file as numpy array
    sp_shimmer_output = total_data[:, shimmer_column]                          #take the column of data that you would like to analyse (7=xgyro)
    x = total_data[:, 0]                                                       #take the timestamp data in column 0 as x
    timestamps = (x - x[0])*1000                                               #adjust timestamps so it starts at 0 and is in milliseconds
    hertz = len(sp_shimmer_output) / x[-1]                                     #sample rate
    
    return timestamps, sp_shimmer_output, hertz


#------------------------------------------------------------------------------
def median_filter(sig, window_size):
    """
    Input: 
    sig = numpy.ndarray of sensor output 
    window_size = uneven integer
    Output: output_sig = np.ndarray of median filtered signal
    """
    output_sig = sc.signal.medfilt(sig, window_size)                           #apply median filter

    return output_sig


#------------------------------------------------------------------------------
def butter_lowpass_filter(data, cutoff, order):
    """
    Input: 
    data = np.ndarray of median filtered sensor data
    cutoff = float between 0 and 1, closer to 0 means less high freq pass
    order = int, order of Butterworth filter
    """
    b, a = sc.signal.butter(order, cutoff, btype='low', analog=False)          #create lowpass butterworth filter
    y = sc.signal.filtfilt(b, a, data)                                         #apply lowpass filter to data
    
    return y


#------------------------------------------------------------------------------
def correct_mounterr(filtered_signal, filename):
    """
    In some cases the shimmer device was mounted upside down, this function is 
    used to flip this data back up again

    filtered_signal = np.ndarray of median & butterworth filtered signal
    filename = string, .mat
    
    if data is still upside down after applying this function, add the patient-
    number to swapbox_patients
    """
    peaks_idx = sc.signal.find_peaks(filtered_signal)                          #find indices of peaks                             
    peaks = filtered_signal[peaks_idx[0]]                                      #take values at these indices
    first_peak = peaks[0]                                                      #take value of the first peak
    
    if first_peak < 240 and first_peak > 0:                                    #if the value of the first peak is lower than 240 and higher than 0 (found by trial and error)
        filtered_signal = filtered_signal*-1                                   #flip the signal
    
    patientno = int(filename[1:4])                                             #store patientno, If E001LS.mat; patientno = 1
    swapbox_patients = []                                                      #data still upside down after applying swap conditions, add value to box
    if patientno in swapbox_patients:
        filtered_signal = filtered_signal*-1                                   #flip the signal
    
    return filtered_signal


#------------------------------------------------------------------------------
def filt_corr_data(patientfile, ctbu):
    """
    Combining functions 1-5 (above)
    Filter & correct for mounting error of patientdata
    Returns filtered, upside oriented data
    
    Input: patientfle (for example: "E001LS.mat")
    ctbu = float between 0 and 1, closer to 0 means less high freq pass 
    """
    datacolumn = 7                                                             #column of data you want to investigate from shimmer
    window_size_median_filter = 21                                             #window size of median filter
    order_butter = 4
    
    time, xgyro, hz = get_data_patient(patientfile, datacolumn)                #take specific shimmer data from patientfile
    medfilt_xgyro = median_filter(xgyro, window_size_median_filter)            #median filter
    filt_xgyro = butter_lowpass_filter(medfilt_xgyro, ctbu, order_butter)      #butterworth low pass filter
    cor = correct_mounterr(filt_xgyro, patientfile)                            #correct for mountingerror (flipping of signal)
    
    return time, cor, hz


#------------------------------------------------------------------------------
def find_peaks_idx(filtered_data, t, mf_height_msv, mf_height_tohs):
    """
    Take the filtered data and determine the toe-off(to), heel strike(hs) & max swing velocity(msv)
    Input:
    filtered_data = numpy.ndarray of filtered & corrected signal
    t = numpy.ndarray of timestamps
    mf_height_msv = float 1-3, determines search area of msv peaks
    mf_height_tohs = float 0-1, determines search area for to & hs throughs
    
    Output:
    3 lists with max swing velocity indices etc.
    """
    average = np.average(filtered_data)
    s = np.std(filtered_data)
    print('Signal data:')
    print('av:', average)
    print('std:', s)
    print('--------------------------------------------------------------')
    
    #search peaks top of graph at minimum of mf_height_msv * standard deviation away from average
    msv_idx, msv = sc.signal.find_peaks(filtered_data, height=average+(mf_height_msv*s))   
    msv_list = msv_idx.tolist()

    #search peaks bottom of the graph at minimum of mf_height_tohs * standard deviation away from the average
    tohs_idx, to = sc.signal.find_peaks(-filtered_data, height=average+(mf_height_tohs*s)) 
    
    all_max = msv_idx.tolist() + tohs_idx.tolist()                             #put indices of all peaks and throughs together                       
    all_max = sorted(all_max)                                                  #sort the indices in ascending order

    to_list = []                                                               #create list for toe off indices
    hs_list = []                                                               #create list for heel strike indices
    for i in range(len(msv_list)):                                             #as long as there are msv peaks
        idx = all_max.index(msv_list[i])
        if idx == 0:                                                           #is an msv peak is found before a through
            hs_list.append(all_max[idx+1])                                     #minima on right to msv max is heel strike
        elif idx != len(all_max)-1:                                            #is the msv peak is not at the end of the list
            to_list.append(all_max[idx-1])                                     #add through before this peak to toe off indices
            hs_list.append(all_max[idx+1])                                     #add through after this peak to heel strike indices
        else:                                                                  #if msv peak is at end of the list
            to_list.append(all_max[idx-1])                                     #add through before this peak to toe off indices
    
    #find timestamps for index of peaks
    msv_values = []                                                            #create list for max swing velocity timestamps
    to_values = []                                                             #create list for toe off timestamps
    hs_values = []                                                             #create list for heel strike timestamps
    for index in msv_list:                                                     #for all msv peaks                                   
        msv_values.append(t[index])                                            #search timestamp and add to msv_value

    for index in to_list:                                                      #for all toe off throughs
        to_values.append(t[index])                                             #search timestamp and add to to_values
    
    for index in hs_list:                                                      #for all heel strike throughs
        hs_values.append(t[index])                                             #search timestamp and add to hs_values
        
    return msv_values, to_values, hs_values


#------------------------------------------------------------------------------
def plot_peaks(data, ti, msvt, tot, hst):
    """
    Plots the signal over time and marks the peaks & throughs
    
    data = numpy.ndarray of filtered & corrected sensordata
    ti = numpy.ndarray of timestamps
    msvt, tot & hst = lists of peak & through indices
    """
    plt.figure(figsize=(15,4))                                                 #set size of figure
    plt.xlabel("Time (ms)")                                                    #name x axis
    plt.ylabel("Angular velocity (deg/s)")                                     #name y axis
    plt.plot(ti, data)                                                         #plot data (angular velocity) vs time
    
    #finding nearest timepoint in angular velocity data to the peaks/throughs found
    #this is needed because the peaks/throughs don't perfectly match the angular velocity data further in the decimals
    #index of this point is taken and used for plotting
    t1 = []

    for value in msvt:
        q = find_nearest(ti, value)                                            #apply function, q = timestamp of peak
        z = np.where(ti==q)                                                    #find index of this peak 
        g = z[0]                                                               #take list with index value from tuple z
        t1.append(g[0])                                                        #take value of index and append to t1
    
    t2 = []
    for value in tot:
        q = find_nearest(ti, value)                                            #apply function, q = timestamp of peak
        z = np.where(ti==q)                                                    #find index of this peak 
        g = z[0]                                                               #take list with index value from tuple z
        t2.append(g[0])                                                        #take value of index and append to t1
    
    t3 = []
    for value in hst:
        q = find_nearest(ti, value)                                            #apply function, q = timestamp of peak
        z = np.where(ti==q)                                                    #find index of this peak 
        g = z[0]                                                               #take list with index value from tuple z
        t3.append(g[0])                                                        #take value of index and append to t1
    
    
    plt.plot(msvt, data[t1], "rx", 
             markeredgewidth=3, label='max swing velocity')                    #put an x on all the msv maxima
    plt.plot(tot, data[t2], "go", markeredgewidth=3, label='toe off')          #put an o on all the toe off maxima
    plt.plot(hst, data[t3], "m+", markeredgewidth=3, label='heel strike')      #put an + on all the heel strike maxima
    plt.legend()
    return 

#------------------------------------------------------------------------------  
def find_nearest(array, value):
    array = np.asarray(array)                                                  #load array as numpy array
    idx = (np.abs(array - value)).argmin()                                     #take absolute value of difference with the searched value, search the index of the one with smallest difference and name idx
    
    return array[idx]                                                          #return value at this idx


#------------------------------------------------------------------------------
def sss_det(dbox, tmsv, tto, ths):
    """
    Takes the indices of the peaks and returns the swing, stride and stance interval
    Input: 
    dbox = numpy.ndarray of filtered & corrected signal
    tmsv, tto & ths = lists of timestamps of peaks & throughs
    
    Output:
    3 lists with all swing, stance & stride intervals (in ms)
    """
    swing = []                                                                 #create list
    stance = []                                                                #create lsit
    stride = []                                                                #create list
    
    total_peaks = tmsv + tto + ths                                             #add all the peaks & troughs into 1 list
    tot = sorted(total_peaks)                                                  #sort list in ascending order
    
    for i in range(len(tmsv)-1):                                               #as long as there are msv peaks
        j = tot.index(tmsv[i])                                                 #find the index of a peak in the total list
        stride.append(tmsv[i+1]-tmsv[i])                                       #stride: time dif between two peaks; add to the list
        swing.append(tot[j+1] - tot[j-1])                                      #swing: time dif between toe off before and heel strike after msv peak; add to list 
        stance.append(tot[j+2] - tot[j+1])                                     #stance: time dif between toe off after and heel strike after the msv peak; add to list
        
    return swing, stance, stride
    

#------------------------------------------------------------------------------
def average_sss(sw, sta, st):
    """
    Calculates average swing, stance & stride time
    All values < 0 and more than 2*std from average are deleted
    
    Input: 3 lists of swing, stance & stride times (ms)
    Output: list; 3 integers: swing average, stance average, stride average
    """
    bb = [sw, sta, st]                                                         #put 3 lists into 1 list
    cor_av_list = []                                                           #create list
    
    for i in range(len(bb)):                                                   #for each of the lists (swing times etc.)
        a = bb[i]                                                              #name list 'a'
        dat = np.asarray(a)                                                    #transform to numpy array and name 'dat'
        dat = dat[dat>0]                                                       #take only positive swing, stride of stance times (negative means peak at beginning - peak at end = huge negative number)
        avr = np.average(dat)                                                  #take average value
        lowl = avr - stdev(dat)*2                                              #lower limit = average - 2* standard deviation
        highl = avr + stdev(dat)*2                                             #higher limit = average + 2* standard deviation
        c1 = dat[dat>lowl]                                                     #take only values higher than lower limit and name c1
        c2 = c1[c1<highl]                                                      #take only values lower than higher limit in c1 and name c2
        cor_avr = np.average(c2)/1000                                          #take average (corrected now) in seconds
        cor_av_list.append(cor_avr)                                            #append average to list 
    
    return cor_av_list
    

#------------------------------------------------------------------------------
def get_sss(ptfile_mat, cutff, multf_msv, multf_tohs):
    """
    Combining functions 1-11
    
    Input:
    ptfile_mat = matlab file with Shimmmer 3 data
    cutff = float between 0 and 1, cutoff low pass filter; 
    multf_msv = float, mult factor of search height of msv peaks
    multf_tohs = float, mult factor of search height of to & hs peaks
    
    Output:
    dbox with average swing, stride and stance
    """
    t, sg, h = filt_corr_data(ptfile_mat, cutff)                               #take filtered and corrected (mounterror) timearray, signalarray and hertz from matlabfile with cutoff value butterworth filter as input
    id_ms, id_to, id_hs = find_peaks_idx(sg, t, multf_msv, multf_tohs)         #find the indices of max swing velocity, toe off and heel strike
    
    swing_times, stance_times, stride_times = sss_det(sg, id_ms, id_to, id_hs) #determine all swing, stride and stance times with these indices
    dbox_av_sss = average_sss(swing_times, stance_times, stride_times)         #filter out values > 2*std and take average (thus 3 in total)

    plot_peaks(sg, t, id_ms, id_to, id_hs)                                     #plot the signal with timestamps and put markers on maxima/minima
    
    c = complexity_index(ptfile_mat, cutff, 2, 0.15, 40)
    #this part is used for automatic saving of the data when running the code
    patientno = int(ptfile_mat[1:4])                                           #find patientnumber, if file "E001LS.mat", patientno = 1
    swi, stan, stri = dbox_av_sss                                              #take averages out of dbox

    return dbox_av_sss


#------------------------------------------------------------------------------
def complexity_index(matlab, hzlim, samplength, tol, maxsc):
    """
    Calculate complexity index of time series
    """
    
    ti, sig, hertz = get_data_patient(matlab, 7)
    sethz = hzlim
    walktime = ti[-1] /1000
    dpoints = int(sethz*walktime)
    resampsig = sc.signal.resample(sig, dpoints)
    plt.plot(resampsig)
    entropy = multiscale_entropy(resampsig, samplength, tol, maxsc)
    ent = entropy[entropy<1000] #remove infinity outcomes
    ci = np.trapz(ent)
    
    return ci


#------------------------------------------------------------------------------
matlab = 'E001LS.mat'
s = 2
r = 0.2
m = 40
hz = 200

c = complexity_index(matlab, hz, s, r, m)
patientno = int(matlab[1:4])

with open('complexity_index_init_left.csv', 'a+') as file:   	                           #open a vsc file with particular name (variable 1 after open), 'a+' determines that if this file is non-existent, create one
    
    #file.write(str(["Patientno","hz", "samplength","tolerance", "maxsc", "Complexity index"]))           #ONLY FIRST PATIENT YOU TEST, first row of csv file will name your variables
    file.write("\n")                                                       #enter
    file.write(str([patientno, hz, s, r, m, c])) 


print(c)

