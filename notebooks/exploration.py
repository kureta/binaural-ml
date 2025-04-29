

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import soundfile
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.signal import butter, lfilter


    import math
    return np, plt, soundfile


@app.cell
def _(np, soundfile):
    audio, sample_rate = soundfile.read("horn.wav")
    mono = audio  # [:, 0]
    mono = mono / np.max(np.abs(mono))
    sample_rate
    return mono, sample_rate


@app.cell
def _(mono, np, sample_rate):
    c = 343  # m/s
    head_radius = 0.09
    samples_per_meter = sample_rate / c  # sample density in space
    distance_between_samples = c / sample_rate
    d = np.arange(0, len(mono)) * distance_between_samples
    return c, head_radius


@app.cell
def _(head_radius, np):
    def fn_elliptic_orbit(t, period=20, rx=50, ry=2):
        f = 1 / period
        return rx * np.cos(f * 2 * np.pi * t), ry * np.sin(f * 2 * np.pi * t)


    def fn_pass_by(t, v=5, x0=-15.5, y_offset=2):
        return v * t + x0, y_offset


    def to_polar(position):
        return np.hypot(*position), np.arctan2(*position[::-1])


    def to_polar_stereo(position):
        x, y = position
        x_left = x + head_radius
        x_right = x - head_radius
        left = to_polar((x_left, y))
        right = to_polar((x_right, y))

        return left, right
    return fn_pass_by, to_polar, to_polar_stereo


@app.cell
def _(
    c,
    fn_pass_by,
    head_radius,
    mono,
    np,
    sample_rate,
    soundfile,
    to_polar,
    to_polar_stereo,
):
    time = np.arange(0, len(mono)) / sample_rate


    def dopplered_sound(signal, fn_position, origin=(0, 0)):
        x0, y0 = origin
        x, y = fn_position(time)
        r, theta = to_polar((x - x0, y - y0))

        sample_reach_time = time - (r / c)
        dopplered = np.interp(sample_reach_time, time, mono)

        return dopplered


    def simple_ild(r_left, r_right, power=3):
        gain_left = (1 / (r_left + 1e-8)) ** power
        gain_right = (1 / (r_right + 1e-8)) ** power
        return gain_left, gain_right


    from scipy.signal import lfilter_zi


    dopplered_left = dopplered_sound(mono, fn_pass_by, (-head_radius, 0))
    dopplered_right = dopplered_sound(mono, fn_pass_by, (head_radius, 0))

    (r_left, theta_left), (r_right, theta_right) = to_polar_stereo(
        fn_pass_by(time)
    )

    gain_left, gain_right = simple_ild(r_left, r_right, power=2)
    dopplered_left *= gain_left
    dopplered_right *= gain_right

    dopplered = np.vstack((dopplered_left, dopplered_right)).T
    soundfile.write("out.wav", dopplered, sample_rate)
    return (gain_left,)


@app.cell
def _(gain_left, plt):
    plt.plot(gain_left)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
