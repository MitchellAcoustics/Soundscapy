import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal


def spectrogram_2ch(filepath, method="matplotlib", plot=True):
    # TODO: Write docs
    # TODO: Write tests

    # Input sanity
    methodlist = ["scipy", "matplotlib"]
    if method not in methodlist:
        raise ValueError(
            "unknown value for method {}, must be one of {}".format(method, methodlist)
        )
    if not filepath.is_file():
        raise FileNotFoundError("File does not exist.", filepath.absolute())

    data, samplerate = sf.read(filepath)

    ch1 = data[:, 0]
    ch2 = data[:, 1]

    # Plot
    if method == "scipy":
        # Channel 1
        f_1, t_1, Sxx_1 = signal.spectrogram(
            ch1, samplerate, return_onesided=True, scaling="density"
        )

        # Channel 2
        f_2, t_2, Sxx_2 = signal.spectrogram(
            ch2, samplerate, return_onesided=True, scaling="density"
        )

        if plot:
            # To plot directly from scipy results, very slow
            #! for some reason, plotting this is extra slow, but should be good matrix output
            plt.subplot(211)
            plt.pcolormesh(t_1, f_1, Sxx_1, shading="gouraud")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")

            plt.subplot(212)
            plt.pcolormesh(t_2, f_2, Sxx_2, shading="gouraud")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")
            plt.show

        return f_1, t_1, Sxx_1, Sxx_2

    elif method == "matplotlib":
        # Much faster, but presumably uses a different method from scipy
        plt.subplot(211)
        Pxx_1, f_1, t_1, image_axis_1 = plt.specgram(ch1, Fs=samplerate, scale="dB")
        plt.subplot(212)
        Pxx_2, f_2, t_2, image_axis_2 = plt.specgram(ch2, Fs=samplerate, scale="dB")

        plt.xlabel("Time")
        plt.ylabel("Frequency")

        if plot:
            plt.show()

        return f_1, t_1, Pxx_1, Pxx_2


if __name__ == "__main__":
    from pathlib import Path

    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/LA101.wav"
    )

    f, t, Pxx_1, Pxx_2 = spectrogram_2ch(filepath, plot="matplotlib")
