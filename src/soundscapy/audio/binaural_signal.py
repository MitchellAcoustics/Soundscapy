from typing import Tuple, Union

import numpy as np
from acoustics import Signal
from loguru import logger


class BinauralSignal(Signal):
    def __new__(cls, data, fs):
        obj = Signal.__new__(cls, data, fs)
        if obj.ndim == 1:
            obj = obj.reshape(1, -1)
        elif obj.ndim > 2 or (obj.ndim == 2 and obj.shape[0] > 2):
            raise ValueError("BinauralSignal must be either mono or stereo")
        return obj

    @property
    def left(self):
        return Signal(self[0], self.fs)

    @property
    def right(self):
        return Signal(self[-1], self.fs) if self.channels == 2 else self.left

    @classmethod
    def from_wav(cls, filename):
        signal = super().from_wav(filename)
        return cls(signal, signal.fs)

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

    def channel(self, index):
        return Signal(self[index], self.fs)

    def as_mono(self):
        if self.channels == 1:
            return self
        else:
            return Signal(np.mean(self, axis=0), self.fs)
