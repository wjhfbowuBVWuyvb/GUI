import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample, correlate
from scipy.io import wavfile

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
    
    # Resample signals to the same length
    resampled_length = min(len(ref_signal), len(signal))
    ref_signal = resample(ref_signal, resampled_length)
    signal = resample(signal, resampled_length)
    
    # Bandpass filter for both signals
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    ref_signal = butter_bandpass_filter(ref_signal, lowcut=0.5, highcut=50, fs=fs)
    signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=fs)
    
    st.subheader("Preprocessed Signals")
    st.write("Reference Signal:")
    st.line_chart(ref_signal)
    st.write("Uploaded Signal:")
    st.line_chart(signal)
    
    # Compute cross-correlation
    correlation = correlate(signal, ref_signal, mode='valid')
    max_correlation = np.max(correlation)
    normalized_correlation = max_correlation / (np.linalg.norm(signal) * np.linalg.norm(ref_signal))
    
    # Correlation threshold
    correlation_threshold = st.slider("Set Correlation Threshold", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
    if normalized_correlation < correlation_threshold:
        st.warning("The uploaded signal deviates significantly from the reference signal.")
    else:
        st.success("The uploaded signal closely matches the reference signal.")
    
    # Plot cross-correlation
    st.subheader("Cross-Correlation")
    fig, ax = plt.subplots()
    ax.plot(correlation, label="Cross-Correlation")
    ax.set_title("Cross-Correlation Between Signals")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.legend()
    st.pyplot(fig)
