import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import subprocess
from pathlib import Path
from typing import Union


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


def loudness_from_wav(
    wavfile: Union[str, Path],
    ref_file: Union[str, Path] = "",
    ref_level: float = None,
    method: str = "time_varying",
    field_type: str = "F",
    time_skip: float = 0.2,
    relocate_iso_exe: Union[bool, str, Path] = False,
):
    """The psychoacoustic loudness of a wav file, according to ISO 532-1.

    This function provides a wrapper for using the CLI executable provided by ISO 532-1. It can call separate methods for Stationary or Time-varying sounds. 

    From the ISO 532-1 Annex C Standard
    Calls ISO_532-1.exe as a command line application. The wrapper will send the following commands to the executable:

    method == `stationary`
        ISO_532-1 Stationary<F|D><input file>[<ref.file><ref.level>]<time skip value>
            Stationary loudness with given WAVE input file with 32-bit float format for free (F) or diffuse (D) field.
            
            Do not consider first <time skip> seconds for calculating third octave levels, e.g. time_skip=0.2

            If <input file> is in 16-bit integer format, a reference file <ref.file> and a reference level <ref.level>(dB rms) shall be provided

            Specific loudness vs. Bark (average) and total loudness (average) are written into CSV-files.

    method == `time_varying`
        ISO_532-1 Time_varying<F|D><input file>[<ref.file><ref.level>]
            Time-varying loudness with given WAVE input file with 32-bit float format for free (F) or diffuse (D) field.

            Specific loudness vs. Bark and time function as well as loudness vs. time function are written into CSV-files

            If <input file> is in 16-bit integer format, a reference file <ref.file> and a reference level <ref.level>(dBrms) shall be provided.

    Supported sound files are 16-bit integer or 32-bit WAVE files (correct sound pressure values, no normalized data) containing one channel. Sampling rates shall be 32 kHz, 44.1 kHz, or 48 kHz. Stationary signals should be longer that <time skip> seconds (start of signal not used for level calculation).


    Parameters
    ----------
    wavfile : Union[str, Path]
        Relative or absolute path to the wav file to analyse
    ref_file : Union[str, Path], optional
        Relative or absolute path to calibration file. Required if 16-bit wav file, optional for 32-bit, by default ""
    ref_level : float, optional
        dB level of calibration file, by default ""
    method : str, optional
        One of ["time_varying", "stationary"], by default "time_varying"
    field_type : str, optional
        Free (F) or diffuse (D) field, by default "F"
    time_skip : float, optional
        Do not consider first `time_skip` seconds for calculating third octave band levels, by default 0.2
    relocate_iso_exe : Union[bool, str, Path], optional
        In standard use, should not redefine. If having relative import errors, point directly to the .exe location, by default False

    Returns
    -------
    dict
        For stationary method, contains the Loudness (sones) and Loudness level (phon): 
            {"loudness_sone": float, "loudness_lvl_phon": float}
        For time_varying method, contains the N_Max Loudness and N_5 Loudness (sones):
            {"loudness_NMax": float, "loudness_N5": float}

    Raises
    ------
    TypeError
        If the ref_level is not a number (float or int)
    TypeError
        If the time_skip is not a number (float or int)
    ValueError
        Stationary loudness from precalculated levels have their own method
    ValueError
        If method is not recognised
    TypeError
        If wavfile is not a str or Path
    FileNotFoundError
        If wavfile is not found
    TypeError
        If path to .exe (relocate_iso_exe) is not a str or Path
    """

    # Even though these values should logically be floats,
    if ref_level and not isinstance(ref_level, (float, int)):
        raise TypeError("ref_level must be a number.")
    if not isinstance(time_skip, (float, int)):
        raise TypeError("time_skip must be a number")

    # Input sanity!
    methodlist = ["time_varying", "stationary"]
    if method not in methodlist:
        if method in {"stationary_levels", "Stationary_Levels"}:
            raise ValueError(
                "Calculating loudness from pre-calculated levels has its own method! (or it soon will...)"
            )
        else:
            raise ValueError(
                "unknown value for method {}, must be one of {}".format(
                    method, methodlist
                )
            )
    # Check wavfile input
    if isinstance(wavfile, Path):
        wavfile = str(wavfile.absolute())
    if isinstance(ref_file, Path):
        ref_file = str(ref_file.absolute())
    if not isinstance(wavfile, str):
        raise TypeError(
            f".wav file paths should either be a string or a pathlib.Path object. type(wavfile): {type(wavfile)}"
        )

    if not Path(wavfile).is_file():
        raise FileNotFoundError(f"File does not exist. wavfile: {wavfile}")

    # There are some problems with relative paths to the .exe. At the moment, this is handled by allowing the user to define their own location for the path.
    if relocate_iso_exe:
        if isinstance(relocate_iso_exe, Path):
            iso_532_exe = str(relocate_iso_exe.absolute())
        elif not isinstance(relocate_iso_exe, str):
            raise TypeError(
                f"filepath to the .exe whould be a string or a pathlib.Path object. type(relocate_iso_exe: {type(relocate_iso_exe)}"
            )
        iso_532_exe = relocate_iso_exe
    else:
        iso_532_exe = "./bin/ISO_532-1"

    # TODO: Check bit depth of wavfile and decide whether to force a ref_file and ref_level. For now, it just assumes you get it right

    if method == "stationary":
        # Generate CLI command: ISO_532-1 Stationary <F|D> <input file>[<ref.file><ref.level>]<time skip value>
        ts_val = 0.2 if time_skip == "" else time_skip
        command = [
            iso_532_exe,
            "Stationary",
            field_type,
            wavfile,
            ref_file,
            str(ref_level),
            str(time_skip),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        loudness_1, loudness_2 = _parse_iso_stdout(result)
        return {
            "loudness_sone": float(loudness_1),
            "loudness_lvl_phon": float(loudness_2),
        }

    elif method == "time_varying":
        # Generate CLI command: ISO_532-1 Time_varying <F|D> <input file>[<ref.file><ref.level>]
        command = [
            iso_532_exe,
            "Time_varying",
            field_type,
            wavfile,
            ref_file,
            str(ref_level),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        loudness_1, loudness_2 = _parse_iso_stdout(result)
        return {
            "loudness_Nmax": float(loudness_1),
            "loudness_N5": float(loudness_2),
        }


def _parse_iso_stdout(iso_subp_result):
    splits = iso_subp_result.stdout.split()
    # Result values from ISO exe, exactly what metric they are depends on the method
    loudness_1 = splits[6]
    loudness_2 = splits[-1]
    return loudness_1, loudness_2


# TODO: finish functions to parse iso csv
def _capture_iso_csv_output(wav_dir, method):
    """
    In addition to its command line outputs, the ISO exe creates a csv with Specific loudness values and vs time values (depending on the method).
    This method catches those after they have been saved, parses them, then deletes them while returning their contents.
    """
    return None


def parse_stationary_loudness_csv(filepath):
    return None


def parse_time_var_loudness_csv(filepath):
    return None


if __name__ == "__main__":
    from pathlib import Path

    filepath = Path(
        "test/test_DB/OFF_LocationA1_FULL_2020-12-31/OFF_LocationA1_BIN_2020-12-31/LocationA1_WAV/LA101.wav"
    )

    f, t, Pxx_1, Pxx_2 = spectrogram_2ch(filepath, plot="matplotlib")

    # TODO: Turn these into doctest examples
    filepath = Path("test\\iso_532-1\\sine 1kHz 40dB 16bit.wav")
    ref_file = Path("test\\iso_532-1\\calibration signal sine 1kHz 60dB.wav")
    stat_result = loudness_from_wav(
        filepath, ref_file, ref_level=60, method="stationary"
    )
    assert stat_result["loudness_sone"] == 1.0
    assert stat_result["loudness_lvl_phon"] == 40.0

    time_result = loudness_from_wav(
        filepath, ref_file, ref_level=60, method="time_varying"
    )
    assert time_result["loudness_Nmax"] == 1.0
    assert time_result["loudness_N5"] == 1.0
