

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import soundfile
    import numpy as np
    from matplotlib import pyplot as plt


    import math
    import marimo as mo
    return mo, np, soundfile


@app.cell
def _(soundfile):
    audio, sample_rate = soundfile.read("data/horn.wav")
    # Get only the first channel if the audio is multichannel
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    sample_rate
    return audio, sample_rate


@app.cell
def _():
    c = 343  # m/s
    head_radius = 0.09
    return c, head_radius


@app.cell
def _(audio, c, np, sample_rate):
    # Define some orbits for the source
    def fn_elliptic_orbit(t, period=20, rx=50, ry=2):
        f = 1 / period
        return rx * np.cos(f * 2 * np.pi * t), ry * np.sin(f * 2 * np.pi * t)


    def fn_pass_by(t, v=5, x0=-15.5, y_offset=2):
        return v * t + x0, y_offset


    # Cartesian to polar coordinates
    def to_polar(position):
        return np.hypot(*position), np.arctan2(*position[::-1])


    # Calculate the actual time it takes for each sample to reach the listener
    def dopplered_sound(signal, r):
        time = np.arange(0, len(signal)) / sample_rate
        sample_reach_time = time - (r / c)
        # Interpolate audio from proper time to sample time
        dopplered = np.interp(sample_reach_time, time, audio)

        return dopplered


    # Calculate L/R gains from distance
    def simple_ild(r_left, r_right, power=2):
        gain_left = (1 / (r_left + 1e-8)) ** power
        gain_right = (1 / (r_right + 1e-8)) ** power
        return gain_left, gain_right


    # TODO: Apply HRTF
    return dopplered_sound, fn_elliptic_orbit, simple_ild, to_polar


@app.cell
def _(
    audio,
    dopplered_sound,
    fn_elliptic_orbit,
    head_radius,
    mo,
    np,
    sample_rate,
    simple_ild,
    soundfile,
    to_polar,
):
    # Get source x,y coordinates per sample
    time = np.arange(0, len(audio)) / sample_rate
    x, y = fn_elliptic_orbit(time)

    # Convert to polar coordinates centered around left and right ears
    # TODO: angles `theta` will be used for HRTF (head-related transfer function)
    # That will filter audio as it passes through the head to reach the ear on the otherside
    r_left, _ = to_polar((x + head_radius, y))
    r_right, _ = to_polar((x - head_radius, y))

    # Apply doppler shift in the time domain
    dopplered_left = dopplered_sound(audio, r_left)
    dopplered_right = dopplered_sound(audio, r_right)

    # Apply gains according to distance
    gain_left, gain_right = simple_ild(r_left, r_right)
    dopplered_left *= gain_left
    dopplered_right *= gain_right

    # Stack channels into a stereo sound
    dopplered = np.vstack((dopplered_left, dopplered_right))
    soundfile.write("out/out.wav", dopplered.T, sample_rate)

    mo.audio(dopplered, rate=sample_rate, normalize=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
