import numpy as np
import av
from scipy.signal import butter, sosfiltfilt

TARGET_SR = 44100


def extract_audio_waveform(video_path, target_sr=TARGET_SR):
    """
    Decode the audio track of video_path and return a mono float32 waveform
    normalized to [-1, 1], resampled to target_sr. Downmixing and resampling
    are done by PyAV's resampler so channel-layout quirks (planar/packed,
    mono/stereo) are handled consistently.
    """

    container = av.open(video_path)

    if not container.streams.audio:
        container.close()
        raise ValueError(f"No audio stream in {video_path}")

    astream = container.streams.audio[0]
    resampler = av.AudioResampler(format='s16', layout='mono', rate=target_sr)

    chunks = []
    for frame in container.decode(astream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray().reshape(-1)
            chunks.append(arr)

    for resampled in resampler.resample(None):  # flush
        arr = resampled.to_ndarray().reshape(-1)
        chunks.append(arr)

    container.close()

    waveform = np.concatenate(chunks).astype(np.float32) / 32768.0

    return waveform, target_sr


def bandpass_energy(waveform, sample_rate, low_hz, high_hz, n_frames, fps=25):
    """
    Band-pass filter the waveform and return per-video-frame RMS energy,
    aligned so that index i covers the audio spanning video frame i.
    """

    samples_per_frame = sample_rate / fps
    total_samples_needed = int(round(n_frames * samples_per_frame))

    if len(waveform) < total_samples_needed:
        waveform = np.pad(waveform, (0, total_samples_needed - len(waveform)))
    else:
        waveform = waveform[:total_samples_needed]

    nyquist = sample_rate / 2
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    sos = butter(4, [low, high], btype='band', output='sos')
    filtered = sosfiltfilt(sos, waveform)

    frame_bounds = (np.arange(n_frames + 1) * samples_per_frame).round().astype(np.int64)
    energy = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = filtered[frame_bounds[i]:frame_bounds[i + 1]]
        energy[i] = np.sqrt(np.mean(seg ** 2)) if seg.size > 0 else 0.0

    return energy


def compute_audio_features(video_path, n_frames, fps=25,
                            crowd_band=(20, 500), commentary_band=(300, 3400)):
    """
    Extract per-frame crowd-noise and commentary energy envelopes for a match.
    Returns dict with 'crowd_energy' and 'commentary_energy', each (n_frames,) float32.
    """

    waveform, sr = extract_audio_waveform(video_path)

    crowd_energy = bandpass_energy(waveform, sr, *crowd_band, n_frames, fps)
    commentary_energy = bandpass_energy(waveform, sr, *commentary_band, n_frames, fps)

    return {
        'crowd_energy': crowd_energy,
        'commentary_energy': commentary_energy,
    }
