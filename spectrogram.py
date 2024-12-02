from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft
import pathlib
from scipy.signal import get_window

# Function to create and save spectrograms
def plot_spectrogram(file_path, save_dir, save_dir_no_axis, fft_sample, window_function, duration, hop_length, start_time, min_freq, max_freq):
    
    sr, audio_signal = wavfile.read(file_path)
    
    # For stereo audio, use only one channel
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal[:, 0]
        
    # Extract frames of specified duration starting from start_time
    start_sample = int(start_time * sr)
    frame = audio_signal[start_sample:start_sample + int(duration * sr)]
    
    if len(frame) < int(duration * sr):
        return  # Skip if the frame is too short
    
    # Perform FFT and apply overlap for each frame
    stft_result = []
    
    for i in range(0, len(frame) - fft_sample, hop_length):
        windowed_frame = frame[i*hop_length:i*hop_length + fft_sample] 

        # Apply the window function
        N = len(windowed_frame)
        if window_function == 'hanning':
            window = np.hanning(fft_sample)
        elif window_function == 'hamming':
            window = np.hamming(fft_sample)
        elif window_function == 'rectangular':
            def rectangular_window(windowed_frame):
                return windowed_frame
            window = rectangular_window(windowed_frame)
        else:
            raise ValueError("Invalid window function. Please choose from 'hanning', 'hamming', or 'rectangular'.")

        input_data = windowed_frame * window
        spectrum = fft(input_data)
        amplitude = np.abs(spectrum)
        amplitude = 1 / (np.sum(window) / N) * amplitude
        # Extract only positive frequency components and add to the list
        stft_result.append(amplitude[:len(amplitude) // 2 + 1])
   
    if stft_result:
        stft_result = np.array(stft_result).T
    else:
        raise ValueError("STFT result is empty. Check your FFT or frame settings.")
    
    # Frequency axis for FFT results
    freqs = np.linspace(0, sr / 2, fft_sample // 2 + 1, endpoint=True)
    
    # Filter based on min_freq and max_freq
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    filtered_spectrogram = stft_result[freq_mask, :]

    # Normalize amplitude
    normalized_spectrogram = (filtered_spectrogram - np.min(filtered_spectrogram)) / (np.max(filtered_spectrogram) - np.min(filtered_spectrogram))

    # Plot spectrogram (with axes and title)
    plt.figure(figsize=(6, 6))
    img = plt.imshow(normalized_spectrogram, aspect='auto', cmap='jet', origin='lower', extent=[0, duration, min_freq, max_freq], vmin=0, vmax=1)
    plt.title(f'Spectrogram of {os.path.basename(file_path)} at {start_time:.1f}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([min_freq, max_freq])
    
    # Add a color bar (amplitude intensity)
    cbar = plt.colorbar(img)
    cbar.set_label('')

    # Path to save the file
    save_path = os.path.join(save_dir, f"{os.path.basename(file_path).split('.')[0]}_{int(start_time * 1000):05d}.png")
    plt.savefig(save_path)  # Save the image
    plt.close()  # Close the plot

    # Plot spectrogram (without axes and title)
    plt.figure(figsize=(6, 6))
    plt.imshow(normalized_spectrogram, aspect='auto', cmap='jet', origin='lower', extent=[0, duration, min_freq, max_freq], vmin=0, vmax=1)
    
    # Path to save the file (no axes or title)
    save_path_no_axis = os.path.join(save_dir_no_axis, f"{os.path.basename(file_path).split('.')[0]}_{int(start_time * 1000):05d}_no_axis.png")
    plt.axis('off')  # Hide axes
    plt.savefig(save_path_no_axis, bbox_inches='tight', pad_inches=0)  # Save the image
    plt.close()  # Close the plot

# Return a pathlib object for the output directory path
# Create the directory if it does not exist
def out_dir(path: str):
    pathlib_dir = pathlib.Path(path)
    if not pathlib_dir.exists():
        pathlib_dir.mkdir(parents=True, exist_ok=True)
    return pathlib_dir

# Main function
def main(dirname_input, samplerate, window_function, fft_sample, overlap, hop_length, fft_number, min_freq, max_freq, plot, plot_number, duration):
    if plot:
        dirname_all_fig = ""
        allfig_dir = out_dir(dirname_all_fig)
        allspec_dir = out_dir(allfig_dir / "")
        spectrum_dir = out_dir(allspec_dir / dirname_input)
        allspec_no_axis_dir = out_dir(allfig_dir / "")  # Directory for no-axis plots
        spentrum_no_axis_dir = out_dir(allspec_no_axis_dir / dirname_input)

    # For each WAV file in the directory
    path_dir = pathlib.Path(dirname_input)
    iter_wav = path_dir.glob("*.wav")
    for filename in iter_wav:
         if filename.suffix.lower() == ".wav":  # Filter only .wav files
            file_path = str(filename)  # Convert pathlib.Path object to string
            for start_time in np.arange(0, 60, duration):
                plot_spectrogram(file_path, spectrum_dir, spentrum_no_axis_dir, fft_sample, window_function, duration, hop_length, start_time, min_freq, max_freq)

                
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
        plot_number=PLOT_NUMBER,
        duration=DURATION)
