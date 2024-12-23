import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile

# Streamlit App Title
st.title("Multi-Channel Signal Processing with Adjustable Parameters")

# File Uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    fs, signal = wavfile.read(uploaded_file)

    # Check the number of channels
    if len(signal.shape) == 1:  # Single-channel signal
        num_channels = 1
        signals = [signal]
    else:  # Multi-channel signal
        num_channels = signal.shape[1]
        signals = [signal[:, i] for i in range(num_channels)]

    st.write(f"Number of channels detected: {num_channels}")

    # Input Parameters
    M = st.number_input("Downsampling Factor (M)", min_value=None, max_value=None, value=12, step=1)
    lowcut = st.number_input("Low Cutoff Frequency (Hz)", min_value=None, max_value=None, value=10.0, step=1.0)
    highcut = st.number_input("High Cutoff Frequency (Hz)", min_value=None, max_value=None, value=800.0, step=1.0)
    order = st.number_input("Butterworth Filter Order", min_value=None, max_value=None, value=2, step=1)
    window_size = st.number_input("Window Size (samples)", min_value=None, max_value=None, value=50, step=10)
    threshold = st.number_input("Uniform Interval Threshold (samples)", min_value=None, max_value=None, value=(0.02 * fs), step=1.0)
    height = st.number_input("Peak Detection Height", min_value=0.01, max_value=None, value=None, step=0.01)
    min_distance = st.number_input("Minimum Distance Between Peaks (samples)", min_value=None, max_value=None, value=(0.1 * fs), step=1)

    # If multi-channel, compute combined signal (average across channels)
    if num_channels > 1:
        combined_signal = np.mean(signal, axis=1)
        # Plot combined signal
        st.subheader("Combined Signal of All Channels")
        fig_combined, ax_combined = plt.subplots(figsize=(12, 4))
        ax_combined.plot(combined_signal, label="Combined Signal (Mean of All Channels)", color="purple")
        ax_combined.set_title("Combined Signal of All Channels")
        ax_combined.set_xlabel("Samples")
        ax_combined.set_ylabel("Amplitude")
        ax_combined.legend(loc='upper right')
        st.pyplot(fig_combined)

    # Process each channel
    for channel_index, signal in enumerate(signals):
        st.subheader(f"Processing Channel {channel_index + 1}...")

        # Bandpass filter
        b, a = butter(order, [lowcut / (fs/2), highcut / (fs/2)], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        # Downsampling
        down_sampled = filtered_signal[::M]
        fs_downsampled = fs / M

        # Normalize and compute Shannon energy
        normalised_signal = down_sampled / np.max(np.abs(down_sampled))
        shannon_energy = -normalised_signal**2 * np.log(normalised_signal**2 + 1e-10)

        # Lowpass filter for Shannon energy envelope
        lowpass_cutoff = 20
        b_lowpass, a_lowpass = butter(order, lowpass_cutoff / (fs_downsampled/2), btype='low')
        shannon_energy_envelope = filtfilt(b_lowpass, a_lowpass, shannon_energy)

        # Peak detection
        all_peaks, _ = find_peaks(shannon_energy_envelope, height=height, distance=min_distance)

        # Check if there are enough peaks
        if len(all_peaks) < 2:
            st.write(f"Channel {channel_index + 1}: Not enough peaks detected to calculate intervals. Skipping analysis.")
            fig_peaks, ax_peaks = plt.subplots(figsize=(12, 2.3))
            ax_peaks.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_peaks.scatter(all_peaks, shannon_energy_envelope[all_peaks], color='green', label="Detected Peaks")
            ax_peaks.set_title(f"Channel {channel_index + 1}: Detected Peaks (Insufficient for Analysis)")
            ax_peaks.set_xlabel("Samples")
            ax_peaks.set_ylabel("Energy")
            ax_peaks.legend(loc='upper right')
            st.pyplot(fig_peaks)
            continue

        # Compute intervals between consecutive peaks
        intervals = np.diff(all_peaks)

        # Check if intervals are nearly uniform
        if np.max(intervals) - np.min(intervals) < threshold:
            st.write(f"Channel {channel_index + 1}: Uniform intervals detected. Only peaks will be plotted.")
            fig_uniform, ax_uniform = plt.subplots(figsize=(12, 2.3))
            ax_uniform.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_uniform.scatter(all_peaks, shannon_energy_envelope[all_peaks], color='green', label="Detected Peaks")
            ax_uniform.set_title(f"Channel {channel_index + 1}: Detected Peaks Only")
            ax_uniform.set_xlabel("Samples")
            ax_uniform.set_ylabel("Energy")
            ax_uniform.legend(loc='upper right')
            st.pyplot(fig_uniform)
        else:
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

            # --- Plot 1: Systolic and Diastolic Rhythm ---
            fig_rhythm, ax_rhythm = plt.subplots(figsize=(12, 2.3))
            ax_rhythm.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_rhythm.scatter(S1_peaks, shannon_energy_envelope[S1_peaks], color='blue', label="S1 Peaks")
            ax_rhythm.scatter(S2_peaks, shannon_energy_envelope[S2_peaks], color='red', label="S2 Peaks")

            for i in range(len(S2_peaks)):
                if i < len(S1_peaks):
                    if S2_peaks[i] < S1_peaks[i]:
                        ax_rhythm.axvspan(S2_peaks[i], S1_peaks[i], color='yellow', alpha=0.3, label="Diastole")
                    else:
                        ax_rhythm.axvspan(S1_peaks[i], S2_peaks[i], color='orange', alpha=0.3, label="Systole")

            ax_rhythm.set_title(f"Channel {channel_index + 1}: Systolic and Diastolic Rhythm")
            ax_rhythm.set_xlabel("Samples")
            ax_rhythm.set_ylabel("Energy")
            ax_rhythm.legend(loc='upper right')
            st.pyplot(fig_rhythm)
