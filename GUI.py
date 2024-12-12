import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# Streamlit App
st.title("Heart Signal Comparison and Cleaning")

# File Uploads
reference_file = st.file_uploader("Upload the Reference Heart Signal (perfect condition)", type="wav")
uploaded_file = st.file_uploader("Upload a New Heart Signal to Compare", type="wav")

if reference_file and uploaded_file:
    # Load and preprocess the reference signal
    ref_fs, ref_signal = wavfile.read(reference_file)
    ref_signal = ref_signal / np.max(np.abs(ref_signal))  # Normalize
    
    # Load and preprocess the uploaded signal
    fs, signal = wavfile.read(uploaded_file)
    signal = signal / np.max(np.abs(signal))  # Normalize
    
    # Bandpass filter for both signals
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    ref_signal = butter_bandpass_filter(ref_signal, lowcut=0.5, highcut=50, fs=ref_fs)
    signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=fs)
    
    st.subheader("Preprocessed Signals")
    st.write("Reference Signal:")
    st.line_chart(ref_signal)
    st.write("Uploaded Signal:")
    st.line_chart(signal)
    
    # Shannon Energy Calculation
    def calculate_shannon_energy(signal):
        normalized_signal = signal / np.max(np.abs(signal))
        return -normalized_signal**2 * np.log(normalized_signal**2 + 1e-10)
    
    ref_energy = calculate_shannon_energy(ref_signal)
    energy = calculate_shannon_energy(signal)
    
    # Compare signals using Dynamic Time Warping
    distance, path = fastdtw(ref_energy, energy, dist=euclidean)
    st.write(f"Dynamic Time Warping Distance: {distance:.2f}")
    
    # Threshold for similarity
    similarity_threshold = st.slider("Set Similarity Threshold", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)
    
    if distance > similarity_threshold:
        st.warning("The uploaded signal deviates significantly from the reference signal.")
    else:
        st.success("The uploaded signal closely matches the reference signal.")
    
    # Highlight parts of the signal that deviate
    deviation_threshold = st.slider("Set Deviation Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    deviations = np.abs(energy - np.interp(np.arange(len(energy)), np.linspace(0, len(energy), len(ref_energy)), ref_energy))
    abnormal_signal = np.where(deviations > deviation_threshold, signal, 0)
    
    st.subheader("Cleaned Signal (Deviations Removed)")
    fig, ax = plt.subplots()
    ax.plot(signal, label="Original Signal")
    ax.plot(abnormal_signal, label="Cleaned Signal", color="red")
    ax.set_title("Signal with Deviations Removed")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)
