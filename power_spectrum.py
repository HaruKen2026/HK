#!/usr/bin/env python
# coding: utf-8

# Import required libraries
from scipy.io import wavfile
import numpy as np
from scipy.fft import fft
from matplotlib import pyplot as plt
import pathlib
import csv
import pandas as pd

# Function to perform Fourier transform
def fourier_transform(data, window_function):
    # Apply the window function
    N = len(data)
    if window_function == 'hanning':
        window = np.hanning(N)  # Use Hanning window
    elif window_function == 'hamming':
        window = np.hamming(N)  # Use Hamming window
    elif window_function == 'rectangular':
        def rectangular_window(data):
            return data
        window = rectangular_window(data)  # Use rectangular (no windowing)
    else:
        raise ValueError("Invalid window function. Please choose from 'hanning', 'hamming', or 'rectangular'.")
    
    input_data = data * window  # Apply the window to the data
    
    # Perform Fourier transform on the time-domain response
    spectrum = fft(input_data)
    
    # Calculate amplitude
    amplitude = np.abs(spectrum)
    
    # Apply window correction
    amplitude = 1 / (np.sum(window) / N) * amplitude
    
    # Extract only positive frequency components
    half_sample = len(amplitude) // 2
    amplitude = amplitude[:half_sample + 1]
    
    # Compute the power spectrum
    power = amplitude ** 2
    
    return power


# Return a pathlib object for the output directory path
# Create the directory if it does not exist
def out_dir(path: str):
    pathlib_dir = pathlib.Path(path)
    if not pathlib_dir.exists():
        pathlib_dir.mkdir()
    return pathlib_dir

# Class to handle graph plotting
class MakeGraph:
    # Constructor
    def __init__(self, xlabel=None, ylabel=None):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.tick_params(direction="in")
        self.ax1.set_xlabel(xlabel)
        self.ax1.set_ylabel(ylabel)
        self.ax1.set_ylim(0, 1)

    # Add a plot to the graph
    def add_plot(self, x, y):
        self.ax1.plot(x, y)

    # Save the graph to a file
    def save_fig(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)

    # Destructor
    def __del__(self):
        plt.close()
        

# Main function
def main(dirname_input, samplerate, window_function, fft_sample, overlap, hop_length, fft_number, min_freq, max_freq, plot, plot_number):
    
    def make_csv(output_path, **kwargs):
        df = pd.DataFrame(kwargs)
        df.to_csv(output_path, index=False)

    path_dir = pathlib.Path(dirname_input)
    iter_wav = path_dir.glob("*.wav")

    dirname_all_csv = ""
    allcsv_dir = out_dir(dirname_all_csv)
    csv_dir = out_dir(allcsv_dir / "")

    if plot:
        dirname_all_fig = ""
        allfig_dir = out_dir(dirname_all_fig)
        allspec_dir = out_dir(allfig_dir / ")
        spectrum_dir = out_dir(allspec_dir / dirname_input)
        
    write_list = ["Volt"]
        
    csv_path = csv_dir / (dirname_input + '.csv')
    with open(csv_path, 'w') as ofile:
        writer = csv.writer(ofile, lineterminator='\n')
        writer.writerow(write_list)

    for path_wav in iter_wav:
        samplerate, data = wavfile.read(path_wav)  # Load the WAV file
        len_data = len(data)
        
        pow_data = []  # Initialize list to store power spectrum data
        
        for i in range(int(fft_number)):
            index_start = hop_length * i
            index_stop = index_start + fft_sample
            if index_stop > len_data:
                break
            # Calculate the size of the frequency range
            frequency_round = np.linspace(0, samplerate / 2, fft_sample // 2 + 1)
            resolution = samplerate / fft_sample
            index_minfreq = int(min_freq // resolution)
            index_maxfreq = int(max_freq // resolution)
            power = fourier_transform(data[index_start:index_stop], window_function)
            pow_data.append(power)
                   
        # Normalize the power spectrum data for machine learning
        spec_pow_round = []
        for i in range(len(pow_data)):
            spec_pow_round.append((pow_data[i] - np.min(pow_data[i])) / (np.max(pow_data[i]) - np.min(pow_data[i])))

            if plot and i < plot_number:
                # Plot the spectrum graph
                specfig = MakeGraph(
                    xlabel="Frequency [Hz]",
                    ylabel="Power [a.p]",)
                specfig.ax1.set_xlim(min_freq, max_freq)
                specfig.add_plot(frequency_round[index_minfreq:index_maxfreq], spec_pow_round[i][index_minfreq:index_maxfreq])
                specfig.save_fig(
                    spectrum_dir / (path_wav.stem + str(i) + ".png"))
                del specfig

                # Save spectrum data to CSV
                make_csv(spectrum_dir / (path_wav.stem + str(i) + ".csv"), Frequency=frequency_round[index_minfreq:index_maxfreq], Power=spec_pow_round[i][index_minfreq:index_maxfreq])

            def write_csv(data, path):
                write_list = []
                write_list.append(path_wav.name[:2])
                write_list.extend(data)
                with open(path, 'a') as ofile:
                    writer = csv.writer(ofile, lineterminator='\n')
                    writer.writerow(write_list)

            write_csv(spec_pow_round[i][index_minfreq:index_maxfreq], csv_path)


# Entry point of the script
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
