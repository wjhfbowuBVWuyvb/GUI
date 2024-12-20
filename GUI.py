# Import required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile

# Streamlit App Configuration
st.title("Heart Sound Signal Analysis")
st.write("Upload a WAV file to analyze its systolic and diastolic rhythms.")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Read the uploaded file
    fs, signal = wavfile.read(uploaded_file)

    # Parameters
    M = 12
    lowcut = 10
    highcut = 800
    order = 2

    # Bandpass filter
    b, a = butter(order, [lowcut / (fs/2), highcut / (fs/2)], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    # Downsampling
    down_sampled = filtered_signal[::M]
    fs = fs / M

    # Normalize and compute Shannon energy
    normalised_signal = down_sampled / np.max(np.abs(down_sampled))
    shannon_energy = -normalised_signal**2 * np.log(normalised_signal**2 + 1e-10)

    # Lowpass filter for Shannon energy envelope
    lowpass_cutoff = 20
    b_lowpass, a_lowpass = butter(order, lowpass_cutoff / (fs/2), btype='low')
    shannon_energy_envelope = filtfilt(b_lowpass, a_lowpass, shannon_energy)

    # Peak detection
    height = 0.1
    min_distance = int(0.1 * fs)
    all_peaks, _ = find_peaks(shannon_energy_envelope, height=height, distance=min_distance)

    # Compute intervals between consecutive peaks
    intervals = np.diff(all_peaks)

    # Classify S1 and S2 peaks dynamically
    S1_peaks = []
    S2_peaks = []

    for i in range(len(intervals)):
        if i == 0:
            if intervals[i] > intervals[i + 1]:
                S2_peaks.append(all_peaks[i])
                S1_peaks.append(all_peaks[i + 1])
            else:
                S1_peaks.append(all_peaks[i])
                S2_peaks.append(all_peaks[i + 1])
        else:
            if intervals[i] > intervals[i - 1]:
                S2_peaks.append(all_peaks[i])
                S1_peaks.append(all_peaks[i + 1])
            else:
                S1_peaks.append(all_peaks[i])
                S2_peaks.append(all_peaks[i + 1])

    S1_peaks = np.array(S1_peaks)
    S2_peaks = np.array(S2_peaks)

    # Define window size for peak extraction
    window_size = 500

    # Extract S1 signal
    s1_signal = np.zeros_like(shannon_energy_envelope)
    for peak in S1_peaks:
        start = max(0, peak - window_size)
        end = min(len(shannon_energy_envelope), peak + window_size)
        s1_signal[start:end] = shannon_energy_envelope[start:end]

    # Extract S2 signal
    s2_signal = np.zeros_like(shannon_energy_envelope)
    for peak in S2_peaks:
        start = max(0, peak - window_size)
        end = min(len(shannon_energy_envelope), peak + window_size)
        s2_signal[start:end] = shannon_energy_envelope[start:end]

    # --- Plot 1: Systolic and Diastolic Rhythm ---
    fig1, ax1 = plt.subplots(figsize=(12, 2.3))
    ax1.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
    ax1.scatter(S1_peaks, shannon_energy_envelope[S1_peaks], color='blue', label="S1 Peaks")
    ax1.scatter(S2_peaks, shannon_energy_envelope[S2_peaks], color='red', label="S2 Peaks")

    diastole_labeled = False
    systole_labeled = False
    for i in range(len(S2_peaks)):
        if i < len(S1_peaks):
            if S2_peaks[i] < S1_peaks[i]:
                ax1.axvspan(S2_peaks[i], S1_peaks[i], color='yellow', alpha=0.3,
                            label="Diastole" if not diastole_labeled else None)
                diastole_labeled = True
            else:
                ax1.axvspan(S1_peaks[i], S2_peaks[i], color='orange', alpha=0.3,
                            label="Systole" if not systole_labeled else None)
                systole_labeled = True
    ax1.set_title("Systolic and Diastolic Rhythm Indicated")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Energy")
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

    # --- Plot 2: S1 Signal ---
    fig2, ax2 = plt.subplots(figsize=(12, 2.3))
    ax2.plot(s1_signal, label="S1 Peaks Signal", color="blue")
    ax2.set_title("Signal with Only S1 Peaks")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Energy")
    ax2.legend(loc='upper right')
    st.pyplot(fig2)

    # --- Plot 3: S2 Signal ---
    fig3, ax3 = plt.subplots(figsize=(12, 2.3))
    ax3.plot(s2_signal, label="S2 Peaks Signal", color="red")
    ax3.set_title("Signal with Only S2 Peaks")
    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Energy")
    ax3.legend(loc='upper right')
    st.pyplot(fig3)

