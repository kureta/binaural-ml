import torch


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(tensor, device=None):
    if device is None:
        device = get_default_device()
    return tensor.to(device)


def pad_audio(audio, target_length):
    pad_len = target_length - audio.shape[-1]
    if pad_len > 0:
        return torch.nn.functional.pad(audio, (0, pad_len))
    else:
        return audio
