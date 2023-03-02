from __future__ import annotations

import abc

import numpy as np
from PyQt5 import QtCore, sip


class AbstractQObject(sip.wrappertype, abc.ABCMeta):
    pass


class IDelay(QtCore.QObject, abc.ABC, metaclass=AbstractQObject):
    move_finished = QtCore.pyqtSignal(bool)
    connection_closed = QtCore.pyqtSignal()
    connection_opened = QtCore.pyqtSignal()

    def __init__(self, time_zero: float):
        super().__init__()
        self._time_zero = time_zero

    @abc.abstractmethod
    def __str__(self) -> str:
        """Display name of the delay hardware"""

    @QtCore.pyqtSlot()
    def initialise(self) -> None:
        self._connect()
        self.connection_opened.emit()

    @property
    def time_zero(self) -> float:
        return self._time_zero

    @time_zero.setter
    def time_zero(self, time_zero: float) -> None:
        self._time_zero = time_zero

    @property
    @abc.abstractmethod
    def tmax(self) -> float:
        """Maximum time point permitted"""

    @property
    @abc.abstractmethod
    def tmin(self) -> float:
        """Maximum time point permitted"""

    @abc.abstractmethod
    def _connect(self) -> None:
        """Connect to and initialise the delay hardware"""

    @abc.abstractmethod
    def _disconnect(self) -> None:
        """Disconnect from the delay hardware"""

    @QtCore.pyqtSlot(float)
    def move_to(self, time_point_ps: float) -> None:
        tflip = self._move(time_point_ps)
        self.move_finished.emit(tflip)

    @QtCore.pyqtSlot()
    @abc.abstractmethod
    def home(self) -> None:
        """Move to the home position"""

    @abc.abstractmethod
    def check_times(self, time_points: np.ndarray) -> bool:
        """Return True if every time in ``times`` lies within the valid range"""

    @abc.abstractmethod
    def check_time(self, time_point: float) -> bool:
        """Return True if ``time`` lies within the valid range"""

    @staticmethod
    def convert_ps_to_mm(time_ps: float) -> float:
        pos_mm = 0.299792458 * time_ps / 2
        return pos_mm

    @staticmethod
    def convert_mm_to_ps(pos_mm: float) -> float:
        time_ps = 2 * pos_mm / 0.299792458
        return time_ps

    @QtCore.pyqtSlot()
    def close(self) -> None:
        self._disconnect()
        self.connection_closed.emit()

    @abc.abstractmethod
    def _move(self, time_point_ps: float) -> bool:
        """Move the delay to the given time.

        Return ``True`` if the delay is large enough come after the next laser shot, in which case the sign of
        the TA signal should be flipped.
        """
