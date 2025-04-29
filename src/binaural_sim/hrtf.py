import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve


# --- Load HRIR Data ---
mat = loadmat("data/hrir_final.mat")
hrir_l_all = mat["hrir_l"]  # shape: (azim, elev, samples)
hrir_r_all = mat["hrir_r"]
azimuths = mat["azimuths"][0]  # azimuths in degrees
elevations = mat["elevations"][0]  # elevations in degrees

# --- Extract Horizontal Plane (elevation = 0°) ---
elev_idx = np.where(elevations == 0)[0][0]
hrir_l = hrir_l_all[:, elev_idx, :]  # shape: (azimuths, samples)
hrir_r = hrir_r_all[:, elev_idx, :]


# --- Interpolation Function ---
def interpolate_hrir(azimuth_deg, az_table, hrir_table):
    # Wrap or clip azimuth to range
    azimuth_deg = np.clip(azimuth_deg, az_table[0], az_table[-1])
    idx_above = np.searchsorted(az_table, azimuth_deg)
    idx_above = np.clip(idx_above, 1, len(az_table) - 1)
    idx_below = idx_above - 1

    az1 = az_table[idx_below]
    az2 = az_table[idx_above]
    w = (azimuth_deg - az1) / (az2 - az1) if az1 != az2 else 0.0

    hrir1 = hrir_table[idx_below]
    hrir2 = hrir_table[idx_above]
    return (1 - w) * hrir1 + w * hrir2


# --- Convolution Function ---
def hrtf_convolve_continuous(
    signal, azimuths_rad, sample_rate, block_size=1024, overlap=0.5
):
    hop = int(block_size * (1 - overlap))
    window = np.hanning(block_size)
    out_L = np.zeros(len(signal) + block_size)
    out_R = np.zeros(len(signal) + block_size)
    win_sum = np.zeros(len(signal) + block_size)

    for i in range(0, len(signal) - block_size, hop):
        frame = signal[i : i + block_size] * window
        az_rad = np.mean(azimuths_rad[i : i + block_size])
        az_deg = az_rad * 180 / np.pi

        h_l = interpolate_hrir(az_deg, azimuths, hrir_l)
        h_r = interpolate_hrir(az_deg, azimuths, hrir_r)

        conv_l = fftconvolve(frame, h_l)[:block_size]
        conv_r = fftconvolve(frame, h_r)[:block_size]

        out_L[i : i + block_size] += conv_l
        out_R[i : i + block_size] += conv_r
        win_sum[i : i + block_size] += window

    win_sum[win_sum == 0] = 1
    stereo = np.vstack((out_L[: len(signal)], out_R[: len(signal)])).T
    return stereo / win_sum[: len(signal), None]


# --- Ready for use ---
# Inputs required:
# `signal`: mono audio signal (1D numpy array)
# `azimuths_rad`: per-sample azimuth angles in radians (same length as `signal`)
# Output:
# stereo signal with HRTF spatialization applied
