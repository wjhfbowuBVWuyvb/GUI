import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile

# Streamlit App
st.title("Heart Signal Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a WAV file of the heart signal", type="wav")
if uploaded_file is not None:
    fs, signal = wavfile.read(uploaded_file)
    st.write(f"Sampling Rate: {fs} Hz")

    # Preprocessing with Butterworth Bandpass Filter (using scipy)
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=fs)

    st.subheader("Preprocessing the Signal")
    st.line_chart(signal)

    # Shannon Energy Calculation
    normalized_signal = signal / np.max(np.abs(signal))
    shannon_energy = -normalized_signal**2 * np.log(normalized_signal**2 + 1e-10)

    # Peak Detection
    st.subheader("Peak Detection")
    height = st.slider("Minimum Peak Height", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    min_distance = st.slider("Minimum Peak Distance (seconds)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    min_distance_samples = int(min_distance * fs)

    peaks, _ = find_peaks(shannon_energy, height=height, distance=min_distance_samples)
    st.write(f"Detected {len(peaks)} peaks.")

    # Plot Signal and Peaks
    fig, ax = plt.subplots()
    ax.plot(shannon_energy, label="Shannon Energy")
    ax.scatter(peaks, shannon_energy[peaks], color="red", label="Peaks")
    ax.set_title("Shannon Energy and Peaks")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Energy")
    ax.legend()
    st.pyplot(fig)

    # Abnormality Detection Based on Rules
    st.subheader("Abnormality Detection")
    abnormal_segments = []
    normal_segments = []

    # Rule-based detection (example: abnormal if peak intervals are irregular)
    peak_intervals = np.diff(peaks) / fs  # Convert to seconds
    avg_interval = np.mean(peak_intervals)
    interval_threshold = st.slider("Irregular Interval Threshold (seconds)", min_value=0.05, max_value=1.0, value=0.2, step=0.05)

    for i, interval in enumerate(peak_intervals):
        if np.abs(interval - avg_interval) > interval_threshold:
            abnormal_segments.append((peaks[i], peaks[i + 1]))
        else:
            normal_segments.append((peaks[i], peaks[i + 1]))

    st.write(f"Detected {len(abnormal_segments)} abnormal intervals.")

    # Highlight Abnormal Segments
    abnormal_signal = np.zeros_like(signal)
    for start, end in abnormal_segments:
        abnormal_signal[start:end] = signal[start:end]

    fig, ax = plt.subplots()
    ax.plot(signal, label="Original Signal")
    ax.plot(abnormal_signal, label="Abnormal Segments", color="red")
    ax.set_title("Heart Signal with Abnormalities Highlighted")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)
