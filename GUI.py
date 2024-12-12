import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

    # Cap Signal Amplitude to a Normal Range (e.g., -1 to 1)
    def limit_amplitude(data, min_val=-1, max_val=1):
        return np.clip(data, min_val, max_val)

    signal = limit_amplitude(signal / np.max(np.abs(signal)))  # Normalize and limit

    # Preprocessing with Butterworth Bandpass Filter (using scipy)
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    signal = butter_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=fs)

    st.subheader("Preprocessing the Signal")
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(y=signal, mode='lines', name='Preprocessed Signal'))
    st.plotly_chart(fig_signal)
    
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

    # S2 and S1 Peak Identification
    s2_peaks = peaks[::2]
    s1_peaks = peaks[1::2]
    systole = [(s1, s2) for s1, s2 in zip(s1_peaks, s2_peaks[1:])]
    diastole = [(s2, s1) for s2, s1 in zip(s2_peaks, s1_peaks)]

    # Interactive Shannon Energy Plot
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(y=shannon_energy, mode='lines', name='Shannon Energy', line=dict(color='black')))
    fig_energy.add_trace(go.Scatter(x=s2_peaks, y=shannon_energy[s2_peaks], mode='markers', name='S2 Peaks', marker=dict(color='red')))
    fig_energy.add_trace(go.Scatter(x=s1_peaks, y=shannon_energy[s1_peaks], mode='markers', name='S1 Peaks', marker=dict(color='blue')))
    for s1, s2 in systole:
        fig_energy.add_vrect(x0=s1, x1=s2, fillcolor='orange', opacity=0.3, line_width=0, annotation_text='Systole')
    for s2, s1 in diastole:
        fig_energy.add_vrect(x0=s2, x1=s1, fillcolor='yellow', opacity=0.3, line_width=0, annotation_text='Diastole')
    st.plotly_chart(fig_energy)

    # Abnormality Detection Using NeuroKit2
    st.subheader("Abnormality Detection")
    try:
        ecg_analysis = nk.ecg_process(signal, sampling_rate=fs)
        ecg_report = nk.ecg_report(ecg_analysis)
        st.write("Abnormality Detection Report:")
        st.write(ecg_report)
    except Exception as e:
        st.error(f"Error in abnormality detection: {e}")

    # Download Options for Plots
    st.download_button(
        label="Download Shannon Energy Plot",
        data=fig_energy.to_html(),
        file_name="shannon_energy_plot.html",
        mime="text/html"
    )

    st.download_button(
        label="Download Preprocessed Signal Plot",
        data=fig_signal.to_html(),
        file_name="preprocessed_signal_plot.html",
        mime="text/html"
    )
