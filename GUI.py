import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.io import wavfile
from tensorflow.keras.models import load_model

# Streamlit App
st.title("Heart Signal Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a WAV file of the heart signal", type="wav")
if uploaded_file is not None:
    fs, signal = wavfile.read(uploaded_file)
    st.write(f"Sampling Rate: {fs} Hz")

    # Preprocessing with NeuroKit2
    st.subheader("Preprocessing the Signal")
    signal = nk.filter_signal(signal, sampling_rate=fs, lowcut=0.5, highcut=50, method="butterworth")
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

    # Load Pretrained TensorFlow Model (for Abnormality Detection)
    st.subheader("Abnormality Detection")
    model = load_model("heart_signal_abnormality_model.h5")  # Replace with your model file

    # Segment and Predict
    segmented_signal = np.array_split(signal, len(signal) // fs)  # 1-second segments
    predictions = [model.predict(np.expand_dims(seg, axis=0)) for seg in segmented_signal]
    predictions = np.array(predictions).flatten()

    # Display Results
    abnormality_threshold = st.slider("Abnormality Detection Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    abnormal_segments = np.where(predictions > abnormality_threshold)[0]

    st.write(f"Detected {len(abnormal_segments)} abnormal segments.")

    # Highlight Abnormal Segments
    abnormal_signal = np.zeros_like(signal)
    for seg in abnormal_segments:
        start = seg * fs
        end = start + fs
        abnormal_signal[start:end] = signal[start:end]

    fig, ax = plt.subplots()
    ax.plot(signal, label="Original Signal")
    ax.plot(abnormal_signal, label="Abnormal Segments", color="red")
    ax.set_title("Heart Signal with Abnormalities Highlighted")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)
