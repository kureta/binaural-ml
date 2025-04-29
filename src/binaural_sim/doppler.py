import torch.nn.functional as F

speed_of_sound = 343.0  # meters/sec


def apply_doppler(signal, radial_velocity, sample_rate):
    """
    Apply differentiable Doppler effect by resampling signal.
    """
    factor = (speed_of_sound) / (speed_of_sound - radial_velocity)
    if factor <= 0:
        factor = 0.01

    n_samples = int(signal.shape[-1] / factor)
    signal = signal.unsqueeze(0).unsqueeze(0)  # (B,C,T)
    signal_resampled = F.interpolate(
        signal, size=n_samples, mode="linear", align_corners=False
    )
    return signal_resampled.squeeze(0).squeeze(0)
