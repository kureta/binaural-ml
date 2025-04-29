import soundfile
import numpy as np

from spaudiopy import sph, process, sig
import math


def main():
    audio, sr = soundfile.read("flute_music.wav")
    audio = sig.MonoSignal(audio[0], sr)

    b_format = sig.AmbiBSignal(sph.src_to_b(audio, 2 * math.pi, 0), sr)
    print(b_format)
    stereo = process.b_to_stereo(b_format)
    stereo = np.vstack(stereo)

    soundfile.write("out.wav", stereo, sr)


if __name__ == "__main__":
    main()
