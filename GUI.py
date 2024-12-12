import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile
from io import BytesIO

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
    min_height = st.number_input("Minimum Peak Height", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_height = st.number_input("Maximum Peak Height", min_value=min_height, max_value=2.0, value=1.0, step=0.01)
    min_distance = st.number_input("Minimum Peak Distance (seconds)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    min_distance_samples = int(min_distance * fs)

    peaks, _ = find_peaks(shannon_energy, height=(min_height, max_height), distance=min_distance_samples)
    st.write(f"Detected {len(peaks)} peaks.")

    # Plot Signal and Peaks
    fig, ax = plt.subplots()
    ax.plot(shannon_energy, label="Shannon Energy")
    ax.scatter(peaks, shannon_energy[peaks], color="red", label="Peaks")
    ax.set_title("Shannon Energy and Peaks")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Energy")
    ax.legend()

    # Display the plot
    st.pyplot(fig)

    # Download Options for Plots
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Shannon Energy Plot",
        data=buffer,
        file_name="shannon_energy_plot.png",
        mime="image/png"
    )

    # Preprocessed Signal Plot
    fig_signal, ax_signal = plt.subplots()
    ax_signal.plot(signal, label="Preprocessed Signal")
    ax_signal.set_title("Preprocessed Signal")
    ax_signal.set_xlabel("Samples")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.legend()

    # Display the plot
    st.pyplot(fig_signal)

    buffer_signal = BytesIO()
    fig_signal.savefig(buffer_signal, format="png")
    buffer_signal.seek(0)

    st.download_button(
        label="Download Preprocessed Signal Plot",
        data=buffer_signal,
        file_name="preprocessed_signal_plot.png",
        mime="image/png"
    )
