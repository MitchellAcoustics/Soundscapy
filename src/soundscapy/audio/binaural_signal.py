from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from acoustics import Signal
from loguru import logger


class BinauralSignal(Signal):
    def __new__(cls, data, fs, filepath=None):
        obj = Signal.__new__(cls, data, fs)
        obj.file_path = filepath
        obj.fs = fs
        if obj.ndim == 1:
            obj = obj.reshape(1, -1)
        elif obj.ndim > 2 or (obj.ndim == 2 and obj.shape[0] > 2):
            raise ValueError("BinauralSignal must be either mono or stereo")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fs = getattr(obj, "fs", None)
        self.recording = getattr(obj, "recording", None)

    @property
    def left(self):
        return Signal(self[0], self.fs)

    @property
    def right(self):
        return Signal(self[-1], self.fs) if self.channels == 2 else self.left

    def calibrate_to(
        self, decibel: float | List[float] | Tuple[float, float], inplace: bool = False
    ) -> "BinauralSignal":
        """
        Calibrate the binaural signal to predefined Leq/dB levels.

        This method allows calibration of both channels either to the same level
        or to different levels for each channel.

        Args:
            decibel (float | List[float] | Tuple[float, float]): Target calibration value(s) in dB (Leq).
                If a single value is provided, both channels will be calibrated to this level.
                If two values are provided, they will be applied to the left and right channels respectively.
            inplace (bool, optional): If True, modify the signal in place. If False, return a new calibrated signal.
                Defaults to False.

        Returns:
            Binaural: Calibrated Binaural signal. If inplace is True, returns self.

        Raises:
            ValueError: If decibel is not a float, or a list/tuple of two floats.

        Example:
            >>> # xdoctest: +SKIP
            >>> signal = BinauralSignal.from_wav("audio.wav")
            >>> calibrated_signal = signal.calibrate_to([60, 62])  # Calibrate left channel to 60 dB and right to 62 dB
        """
        if isinstance(decibel, (np.ndarray, pd.Series)):  # Force into tuple
            decibel = tuple(decibel)
        if isinstance(decibel, (list, tuple)):
            if len(decibel) == 2:  # Per-channel calibration (recommended)
                decibel = np.array(decibel)
                decibel = decibel[..., None]
                return super().calibrate_to(decibel, inplace)
            elif (
                len(decibel) == 1
            ):  # if one value given in tuple, assume same for both channels
                decibel = decibel[0]
            else:
                raise ValueError(
                    "decibel must either be a single value or a 2 value tuple"
                )
        if isinstance(decibel, (int, float)):  # Calibrate both channels to same value
            return super().calibrate_to(decibel, inplace)
        else:
            raise ValueError("decibel must be a single value or a 2 value tuple")

    @classmethod
    def from_wav(
        cls,
        filename: Path | str,
        calibrate_to: float | List | Tuple = None,
        normalize: bool = False,
    ):
        """Load a wav file and return a Binaural object

        Overrides the Signal.from_wav method to return a
        Binaural object instead of a Signal object.

        Parameters
        ----------
        filename : Path, str
            Filename of wav file to load
        calibrate_to : float, list or tuple of float, optional
            Value(s) to calibrate to in dB (Leq)
            Can also handle np.ndarray and pd.Series of length 2.
            If only one value is passed, will calibrate both channels to the same value.
        normalize : bool, optional
            Whether to normalize the signal, by default False

        Returns
        -------
        Binaural
            Binaural signal object of wav recording

        See Also
        --------
        acoustics.Signal.from_wav : Base method for loading wav files
        """
        s = super().from_wav(filename, normalize)
        if calibrate_to is not None:
            s.calibrate_to(calibrate_to, inplace=True)
        if isinstance(filename, Path):
            filename = str(filename.stem)
        return cls(s, s.fs, filepath=filename)

    @classmethod
    def channel(self, channel):
        """Get a single channel from the signal

        Parameters
        ----------
        channel : int
            Channel to get (0 or 1)

        Returns
        -------
        Signal
            Single channel signal
        """
        if self.channels == 1:
            return self
        elif (
            channel is None
            or channel == "both"
            or channel == ("Left", "Right")
            or channel == ["Left", "Right"]
        ):
            return self
        elif channel in ["Left", 0, "L"]:
            return self[0]
        elif channel in ["Right", 1, "R"]:
            return self[1]
        else:
            logger.warn("Channel not recognised. Returning Binaural object as is.")
            return self

    def to_wav(self, filename, depth=16):
        if self.channels == 1:
            super().to_wav(filename, depth)
        else:
            super(Signal, self).to_wav(filename, depth)

    def calibrate_to(self, decibel: Union[float, Tuple[float, float]], inplace=False):
        if isinstance(decibel, (tuple, list)) and len(decibel) == 2:
            left_calibrated = self.left.calibrate_to(decibel[0], inplace=False)
            right_calibrated = self.right.calibrate_to(decibel[1], inplace=False)
            calibrated = np.vstack((left_calibrated, right_calibrated))
        else:
            calibrated = super().calibrate_to(decibel, inplace=False)

        if inplace:
            self[:] = calibrated
            return self
        else:
            return BinauralSignal(calibrated, self.fs)

    def as_mono(self):
        if self.channels == 1:
            return self
        else:
            return Signal(np.mean(self, axis=0), self.fs)
