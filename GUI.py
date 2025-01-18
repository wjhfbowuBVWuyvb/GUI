import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile
import io

# Streamlit App Title
st.title("GUI")

# File Uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file is not None:
    fs, signal = wavfile.read(uploaded_file)
    st.write(f"Sampling frequency: {fs} Hz")

    # Check the number of channels
    if len(signal.shape) == 1:  # Single-channel signal
        num_channels = 1
        signals = [signal]
    else:  # Multi-channel signal
        num_channels = signal.shape[1]
        signals = [signal[:, i] for i in range(num_channels)]

    st.write(f"Number of channels detected: {num_channels}")
    M = int(fs / 4000)
    multi_channel_s1_signal = []
    multi_channel_s2_signal = []

    # Input Parameters 
    lowcut = st.number_input("Low Cutoff Frequency (Hz)", min_value=None, max_value=None, value=10.0, step=1.0)
    highcut = st.number_input("High Cutoff Frequency (Hz)", min_value=None, max_value=None, value=800.0, step=1.0)
    order = st.number_input("Butterworth Filter Order", min_value=None, max_value=None, value=2, step=1)
    window_size = st.number_input("Window Size for S1 and S2 plots (samples)", min_value=None, max_value=None, value=500, step=10)
    threshold = st.number_input("Uniform Interval Threshold (samples)", min_value=None, max_value=None, value=0.02 * fs, step=1.0)
    height = st.number_input("Peak Detection Height", min_value=None, max_value=None, value=0.1, step=0.01)
    min_distance = st.number_input("Minimum Distance Between Peaks (samples)", min_value=None, max_value=None, value=400, step=1)
    murmur_duration = st.number_input("Murmur Duration(s)", min_value=None, max_value=None, value=int(0.15 * fs / M), step=1)
    snr_threshold = st.number_input("SNR threshold (dB)", min_value=None, max_value=None, value=50, step=1)
    amplitude_threshold = st.number_input("Murmur amplitude threshold", min_value=None, max_value=None, value=int(0.04), step=1)
    exclusion_zone = st.number_input("exclusion zone around S1/S2", min_value=None, max_value=None, value=int(0.05 * fs / M), step=1)
    
    # Select channels to keep
    channels_to_keep = st.multiselect(
        "Select Channels to Keep", options=list(range(1, num_channels + 1)), default=list(range(1, num_channels + 1))
    )

    # Filter out unwanted channels
    if len(channels_to_keep) > 0:
        signals = [signals[i - 1] for i in channels_to_keep]
        st.write(f"Channels kept: {channels_to_keep}")
    else:
        st.warning("No channels selected")
    
    # Plot signal 
    st.subheader("Signal")
    signal_title = st.text_input("Enter title", value="Signal")
    fig_signal, ax_signal = plt.subplots(figsize=(12, 6))
    if num_channels > 1:
        for i, channel_signal in enumerate(signals):
            ax_signal.plot(channel_signal, label=f"Channel {i + 1}")
    else:
        ax_signal.plot(signal, label="Mono Channel")
    signal_length = len(signal) if num_channels == 1 else len(signals[0])
    xlim_start_signal = st.number_input("X-axis Start for Signal Plot", min_value=0, max_value=len(signal), value=0, step=1)
    xlim_end_signal = st.number_input("X-axis End for Signal Plot", min_value=0, max_value=signal_length, value=len(signal), step=1)
    ax_signal.set_xlim([xlim_start_signal, xlim_end_signal])
    ax_signal.set_title(signal_title)  
    ax_signal.set_xlabel("Samples")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.legend(loc='upper right')
    st.pyplot(fig_signal)
    
    # Allow download of the raw signal plot
    signal_image_buffer = io.BytesIO()
    fig_signal.savefig(signal_image_buffer, format='pdf', dpi=300)
    signal_image_buffer.seek(0)
    st.download_button("Download Signal Plot", signal_image_buffer, file_name="signal.pdf")
    
    def compute_snr(signal, fs, noise_band=(2000, 4000)):
        b_signal, a_signal = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
        signal_filtered = filtfilt(b_signal, a_signal, signal)
        signal_power = np.mean(signal_filtered**2)
    
    
        b_noise, a_noise = butter(order, [noise_band[0] / (fs / 2), noise_band[1] / (fs / 2)], btype='band')
        noise_filtered = filtfilt(b_noise, a_noise, signal)
        noise_power = np.mean(noise_filtered**2)
    
    
        snr = 10 * np.log10(signal_power / (noise_power))
        return snr
    # Process each channel
    processed_signals = []
    for channel_index, signal in enumerate(signals):
        st.subheader(f"Processing Channel {channel_index + 1}...")

        try:
            snr = compute_snr(signal, fs)
            st.write(f"SNR for Channel {channel_index + 1}: {snr:.2f} dB")
        except ValueError as e:
            st.write(f"Error in SNR computation: {e}")
            snr = float('inf')

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
            peak_title = st.text_input("Enter title", value=f"Channel {channel_index + 1}: Detected Peaks")
            st.write(f"Channel {channel_index + 1}: Not enough peaks")
            fig_peaks, ax_peaks = plt.subplots(figsize=(12, 6))
            ax_peaks.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_peaks.scatter(all_peaks, shannon_energy_envelope[all_peaks], color='green', label="Detected Peaks")
            signal_length = len(signal)
            xlim_start_peaks = st.number_input(f"X-axis Start for Peaks Plot (Channel {channel_index + 1})", min_value=0, max_value=signal_length, value=0, step=1)
            xlim_end_peaks = st.number_input(f"X-axis End for Peaks Plot (Channel {channel_index + 1})", min_value=0, max_value=signal_length, value=signal_length, step=1)
            ax_peaks.set_xlim([xlim_start_peaks, xlim_end_peaks])
            ax_peaks.set_title(peak_title)
            ax_peaks.set_xlabel("Samples")
            ax_peaks.set_ylabel("Energy")
            ax_peaks.legend(loc='upper right')
            st.pyplot(fig_peaks)

            # Allow download of the detected plot
            peaks_image_buffer = io.BytesIO()
            fig_peaks.savefig(peaks_image_buffer, format='pdf', dpi=300)
            peaks_image_buffer.seek(0)
            st.download_button(
                f"Download Peaks Plot for Channel {channel_index + 1}",
                peaks_image_buffer,
                file_name=f"channel_{channel_index + 1}_peaks.pdf"
            )
            continue

        # Compute intervals between peaks
        intervals = np.diff(all_peaks)

        # Check if intervals are nearly uniform
        if np.max(intervals) - np.min(intervals) < threshold:
            uniform_title = st.text_input("Enter title", value=f"Channel {channel_index + 1}: Detected Peaks Only")
            st.write(f"Channel {channel_index + 1}: Uniform intervals detected")
            fig_uniform, ax_uniform = plt.subplots(figsize=(12, 6))
            ax_uniform.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_uniform.scatter(all_peaks, shannon_energy_envelope[all_peaks], color='green', label="Detected Peaks")
            signal_length = len(signal)
            xlim_start_uniform = st.number_input(f"X-axis Start for Uniform Peaks Plot (Channel {channel_index + 1})", min_value=0, max_value=signal_length, value=0, step=1)
            xlim_end_uniform = st.number_input(f"X-axis End for Uniform Peaks Plot (Channel {channel_index + 1})", min_value=0, max_value=signal_length, value=signal_length, step=1)
            ax_uniform.set_xlim([xlim_start_uniform, xlim_end_uniform])
            ax_uniform.set_title(uniform_title)
            ax_uniform.set_xlabel("Samples")
            ax_uniform.set_ylabel("Energy")
            ax_uniform.legend(loc='upper right')
            st.pyplot(fig_uniform)

            # Allow download of uniform interval plot
            uniform_image_buffer = io.BytesIO()
            fig_uniform.savefig(uniform_image_buffer, format='pdf', dpi=300)
            uniform_image_buffer.seek(0)
            st.download_button(
                f"Download Uniform Peaks Plot for Channel {channel_index + 1}",
                uniform_image_buffer,
                file_name=f"channel_{channel_index + 1}_uniform_peaks.pdf"
            )
        else:
            # Classify S1 and S2 peaks 
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

            S1_peaks = np.array(S1_peaks, dtype=int)
            S2_peaks = np.array(S2_peaks, dtype=int)

            S1_peaks = S1_peaks[S1_peaks < len(shannon_energy_envelope)]
            S2_peaks = S2_peaks[S2_peaks < len(shannon_energy_envelope)]

            S1_peaks_no_gaps = np.unique(S1_peaks)
            S2_peaks_no_gaps = np.unique(S2_peaks)

            s1_signal_no_gaps = []
            s2_signal_no_gaps = []

            # Extract S1 signal
            for peak in S1_peaks_no_gaps:
                start = max(0, peak - window_size)
                end = min(len(shannon_energy_envelope), peak + window_size)
                s1_signal_no_gaps.append(shannon_energy_envelope[start:end])
        
            for peak in S2_peaks_no_gaps:
                start = max(0, peak - window_size)
                end = min(len(shannon_energy_envelope), peak + window_size)
                s2_signal_no_gaps.append(shannon_energy_envelope[start:end])

            s1_signal_no_gaps = np.hstack(s1_signal_no_gaps)
            s2_signal_no_gaps = np.hstack(s2_signal_no_gaps)
        
            multi_channel_s1_signal.append(s1_signal_no_gaps)
            multi_channel_s2_signal.append(s2_signal_no_gaps)
        # Murmur algorithm
            Murmur_intervals = []
            if fs == 4000:
                exclusion_mask = np.zeros_like(shannon_energy_envelope, dtype=bool)
                for peak in np.concatenate([S1_peaks, S2_peaks]):
                    start = max(0, peak - exclusion_zone)
                    end = min(len(shannon_energy_envelope), peak + exclusion_zone)
                    exclusion_mask[start:end] = True
        
        
                murmur_candidates, _ = find_peaks(
                    shannon_energy_envelope, height=amplitude_threshold
                )
                murmur_candidates = [m for m in murmur_candidates if not exclusion_mask[m]]
        
                for murmur in murmur_candidates:
                    Murmur_intervals.append((murmur, murmur + murmur_duration))
            elif snr <= snr_threshold:
                if len(S1_peaks) > 0:
                    pre_systolic_segment = shannon_energy_envelope[:S1_peaks[0]]
                    pre_systolic_murmurs, _ = find_peaks(pre_systolic_segment, height=amplitude_threshold, prominence=0.01)
                    for murmur in pre_systolic_murmurs:
                        Murmur_intervals.append((murmur, murmur + murmur_duration))
        
        
                for i in range(len(S2_peaks)):
                    if i < len(S1_peaks):
                        systole_start = S1_peaks[i]
                        systole_end = S2_peaks[i]
                        systole_murmurs, _ = find_peaks(
                            shannon_energy_envelope[systole_start:systole_end], height=amplitude_threshold, prominence=0.01
                        )
                        systole_murmurs += systole_start
                        Murmur_intervals.extend([(murmur, 'systole') for murmur in systole_murmurs])
        
                    if i + 1 < len(S1_peaks):
                        diastole_start = S2_peaks[i]
                        diastole_end = S1_peaks[i + 1]
                        diastole_murmurs, _ = find_peaks(
                            shannon_energy_envelope[diastole_start:diastole_end], height=amplitude_threshold, prominence=0.01
                        )
                        diastole_murmurs += diastole_start
                        Murmur_intervals.extend([(murmur, 'diastole') for murmur in diastole_murmurs])
        
            Murmur_peaks = [m[0] for m in Murmur_intervals if isinstance(m, tuple)]

            # --- Plot 1: Systolic and Diastolic Rhythm ---
            time_axis_downsampled = np.arange(len(shannon_energy_envelope)) / fs_downsampled
            rhythm_title = st.text_input("Enter title", value=f"Channel {channel_index + 1}: Systolic, Diastolic and Murmur")
            fig_rhythm, ax_rhythm = plt.subplots(figsize=(12, 6))
            ax_rhythm.plot(shannon_energy_envelope, label="Shannon Energy Envelope", color="black")
            ax_rhythm.scatter(S1_peaks, shannon_energy_envelope[S1_peaks], color='blue', label="S1 Peaks")
            ax_rhythm.scatter(S2_peaks, shannon_energy_envelope[S2_peaks], color='red', label="S2 Peaks")
            for murmur in Murmur_peaks:
                start = max(0, murmur - murmur_duration)
                end = min(len(shannon_energy_envelope), murmur + murmur_duration)
                plt.axvspan(start, end, color='green', alpha=0.2, label="Murmur Duration" if murmur == Murmur_peaks[0] else None)
            xlim_start_rhythm = st.number_input(f"X-axis Start for Rhythm Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value=0, step=1)
            xlim_end_rhythm = st.number_input(f"X-axis End for Rhythm Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value=len(down_sampled), step=1)
            ax_rhythm.set_xlim([xlim_start_rhythm/fs, xlim_end_rhythm/fs])

            diastole_labeled = False
            systole_labeled = False

            for i in range(len(S2_peaks)):
                if i < len(S1_peaks):
                    if S2_peaks[i] < S1_peaks[i]:
                        ax_rhythm.axvspan(S2_peaks[i], S1_peaks[i], color='yellow', alpha=0.3, 
                                          label="Diastole" if not diastole_labeled else None)
                        diastole_labeled = True
                    else:
                        ax_rhythm.axvspan(S1_peaks[i], S2_peaks[i], color='orange', alpha=0.3, 
                                          label="Systole" if not systole_labeled else None)
                        systole_labeled = True

            ax_rhythm.set_title(rhythm_title)
            ax_rhythm.set_xlabel("Samples")
            ax_rhythm.set_ylabel("Energy")
            ax_rhythm.legend(loc='upper right')
            st.pyplot(fig_rhythm)

            # Allow download of the systolic and diastolic rhythm plot
            rhythm_image_buffer = io.BytesIO()
            fig_rhythm.savefig(rhythm_image_buffer, format='pdf', dpi=300)
            rhythm_image_buffer.seek(0)
            st.download_button(
                f"Download Systolic and Diastolic Plot for Channel {channel_index + 1}",
                rhythm_image_buffer,
                file_name=f"channel_{channel_index + 1}_systolic_diastolic.pdf"
            )

            # --- Plot 2: S1 Peaks ---
            s1_title = st.text_input("Enter title", value=f"Channel {channel_index + 1}: S1 Peaks")
            fig_s1, ax_s1 = plt.subplots(figsize=(12, 6))
            ax_s1.plot(label="S1 Peaks Signal", color="blue")
            xlim_start_s1 = st.number_input(f"X-axis Start for S1 Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value=0, step=1)
            xlim_end_s1 = st.number_input(f"X-axis End for S1 Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value=len(s1_signal_no_gaps), step=1)
            ax_s1.set_xlim([xlim_start_s1, xlim_end_s1])
            ax_s1.set_title(s1_title)
            ax_s1.set_xlabel("Time[s]")
            ax_s1.set_ylabel("Energy")
            ax_s1.legend(loc='upper right')
            st.pyplot(fig_s1)

            # Allow download of S1 plot
            S1_image = io.BytesIO()
            fig_s1.savefig(S1_image, format='pdf', dpi=300)
            S1_image.seek(0)
            st.download_button(
                f"Download S1 plot for Channel {channel_index + 1}",
                S1_image,
                file_name=f"channel_{channel_index + 1}_S1_plot.pdf"
            )

            # --- Plot 3: S2 Peaks ---
            s2_title = st.text_input("Enter title", value=f"Channel {channel_index + 1}: S2 Peaks")
            fig_s2, ax_s2 = plt.subplots(figsize=(12, 6))
            ax_s2.plot(label="S2 Peaks Signal", color="red")
            xlim_start_s2 = st.number_input(f"X-axis Start for S2 Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value=0, step=1)
            xlim_end_s2 = st.number_input(f"X-axis End for S2 Plot (Channel {channel_index + 1})", min_value=0, max_value=len(signal), value= len(s2_signal_no_gaps), step=1)
            ax_s2.set_xlim([xlim_start_s2, xlim_end_s2])
            ax_s2.set_title(s2_title)
            ax_s2.set_xlabel("Samples")
            ax_s2.set_ylabel("Energy")
            ax_s2.legend(loc='upper right')
            st.pyplot(fig_s2)

            # Allow download of S2 plot
            S2_image = io.BytesIO()
            fig_s2.savefig(S2_image, format='pdf', dpi=300)
            S2_image.seek(0)
            st.download_button(
                f"Download S2 plot for Channel {channel_index + 1}",
                S2_image,
                file_name=f"channel_{channel_index + 1}_S2_plot.pdf"
            )

        # Append processed signals
        processed_signals.append(filtered_signal)

    # Combine remaining signals into a new multi-channel WAV file
    if len(processed_signals) > 1:
        new_signal = np.stack(processed_signals, axis=1)
    else:
        new_signal = processed_signals[0]

    in_length_s1 = min(s.shape[0] for s in multi_channel_s1_signal)
    min_length_s2 = min(s.shape[0] for s in multi_channel_s2_signal)

    
    multi_channel_s1_signal = [s[:min_length_s1] for s in multi_channel_s1_signal]
    multi_channel_s2_signal = [s[:min_length_s2] for s in multi_channel_s2_signal]

    
    final_s1_signal = np.column_stack(multi_channel_s1_signal)
    final_s2_signal = np.column_stack(multi_channel_s2_signal)


    # Save the new WAV file
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, fs, new_signal.astype(np.int16))
    wav_buffer.seek(0)
    st.download_button("Download Processed Signal as WAV", wav_buffer, file_name="processed_signal.wav")

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, int(fs_downsampled), (final_s1_signal * 32767).astype(np.int16))
    wav_buffer.seek(0)
    st.download_button("Download Combined S1 Signals", wav_buffer, file_name="combined_s1.wav")
    
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, int(fs_downsampled), (final_s2_signal * 32767).astype(np.int16))
    wav_buffer.seek(0)
    st.download_button("Download Combined S2 Signals", wav_buffer, file_name="combined_s2.wav")

