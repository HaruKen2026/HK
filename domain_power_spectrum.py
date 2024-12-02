#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.io import wavfile
import numpy as np
from scipy.fft import fft
from matplotlib import pyplot as plt
import pathlib
import csv
import pandas as pd
import seaborn as sns


# Function to perform Fourier transform
def fourier_transform(data, window_function):
    # Apply window function
    N = len(data)
    if window_function == 'hanning':
        window = np.hanning(N) # Use Hanning window
    elif window_function == 'hamming':
        window = np.hamming(N) # Use Hamming window
    elif window_function == 'rectangular':
        def rectangular_window(data):
            return data
        window = rectangular_window(data)
    else:
        raise ValueError("Invalid window function. Please choose from 'hanning', 'hamming', or 'rectangular'.")
    
    input_data = data * window
    
    # Perform Fourier transform on time-domain response
    spectrum = fft(input_data)
    
    # Calculate amplitude
    amplitude = np.abs(spectrum)
    
    # Window correction
    amplitude = 1 / (np.sum(window) / N) * amplitude
    
    # Extract only positive frequency components
    half_sample = len(amplitude) // 2
    amplitude = amplitude[:half_sample + 1]
    
    power = amplitude ** 2
    
    return power


# Return a pathlib object for the output directory path
# Create the directory if it does not exist
def out_dir(path: str):
    pathlib_dir = pathlib.Path(path)
    if not pathlib_dir.exists():
        pathlib_dir.mkdir()
    return pathlib_dir
        

# Main function
def main(dirname_input, samplerate, window_function, fft_sample, overlap, hop_length, fft_number, min_freq, max_freq, plot, plot_number):
    
    path_dir = pathlib.Path(dirname_input)
    iter_wav = path_dir.glob("*.wav")
    
    if plot:
        dirname_all_fig = ""
        allfig_dir = out_dir(dirname_all_fig)
        allspec_dir = out_dir(allfig_dir / "")
        spectrum_dir = out_dir(allspec_dir / dirname_input)

    for path_wav in iter_wav:
        samplerate, data = wavfile.read(path_wav)
        len_data = len(data)
        
        pow_data = []
        
        for i in range(int(fft_number)):
            index_start = hop_length * i
            index_stop = index_start + fft_sample
            if index_stop > len_data:
                break
            # Calculate the frequency axis
            resolution = samplerate / fft_sample
            index_minfreq = int(min_freq // resolution)
            index_maxfreq = int(max_freq // resolution)
            pow = fourier_transform(data[index_start:index_stop], window_function)
            # Filter amplitude data based on min_freq and max_freq
            filtered_pow = pow[index_minfreq:index_maxfreq]
            pow_data.append(filtered_pow)
                   
        
        # Min-max normalization for machine learning
        spec_pow_round = []
        for i in range(len(pow_data)):
            spec_pow_round.append((pow_data[i] - np.min(pow_data[i])) / (np.max(pow_data[i]) - np.min(pow_data[i])))

        
        # Number of compressed data points (vertical pixels in the image)
        data_size = 224
        
        # Compressed data
        # Extract maximum values
        power_data_normal_comp = []
        for i in range(len(spec_pow_round)):
            power_data_normal_comp_pre = []
            # Get the size of the filtered data
            filtered_length = len(spec_pow_round[i])
            for j in range(data_size):
                start = int(filtered_length // data_size) * j
                finish = int(filtered_length // data_size) * (j + 1)
                power_data_normal_comp_pre.append(np.max(spec_pow_round[i][start:finish]))
            power_data_normal_comp.append(power_data_normal_comp_pre)
        
        # Convert to a 2D image
        for i in range(len(spec_pow_round)):
            picture = [[255] * data_size for _ in range(data_size)]  # Initialize once
            for j in range(data_size):
                for k in range(data_size):
                    if (power_data_normal_comp[i][j] > (1 / data_size) * k):
                        picture[223 - k][j] = 0
                        
            dpi = 300
            fig = plt.figure(figsize=(224 / dpi, 224 / dpi), dpi=dpi)
            sns.heatmap(picture, cmap='Greys', cbar=None, square=True)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.box(False)
            
            # Save the image
            if plot:
                fig_name = f"{path_wav.stem}_{i}.png"  # Name of the image file
                fig_path = spectrum_dir / fig_name
                plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
            

if __name__ == "__main__":
    main(
        dirname_input=DIRNAME_INPUT,
        samplerate=SAMPLERATE,
        window_function=WINDOW_FUNCTION,
        fft_sample=FFT_SAMPLE,
        overlap=OVERLAP,
        hop_length=HOP_LENGTH,
        fft_number=FFT_NUMBER,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        plot=PLOT,
        plot_number=PLOT_NUMBER)
