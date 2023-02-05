#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:11:04 2023

@author: Silvana
"""
import os  
import pandas as pd
import numpy as np
import scipy.io.wavfile
import scipy.signal as sps

# load dataset names
file_csv = pd.read_csv('esc50-dataset/esc50.csv')
names = file_csv['filename']

# parameters
# downsampling to 8kHz
new_rate = 8000

# parameters for the Mel spectrum    
pre_emphasis = 0.97
# parameters for framing, 25 ms window, 10 ms steps
frame_size = 0.025 
frame_stride = 0.01
# Number of FFT points
NFFT = 5012
# number of triangular filters
nfilt = 128

# =================== PREPROCESSING ===================
for ind in range(len(names)):
    sample_rate, signal_tmp = scipy.io.wavfile.read('esc50-dataset/audio/'+names[ind])
    # compute new length for the signal
    number_of_samples = round(len(signal_tmp) * float(new_rate) / sample_rate)
    # downsampling    
    signal = sps.resample(signal_tmp, number_of_samples)
    emph_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    sample_rate = new_rate
    # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
    signal_length = len(emph_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples 
    pad_signal = np.append(emph_signal, z) 
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Hamming window multiplication
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    # MEL filterbanks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    fbanks = np.dot(pow_frames, fbank.T)
    fbanks = np.where(fbanks == 0, np.finfo(float).eps, fbanks)  # Numerical Stability
    fbanks = 20 * np.log10(fbanks)  # dB
    # save data individually in folder Preprocessed
    np.savez('Preprocessed/' +names[ind]+ '.npz',fbanks=fbanks)