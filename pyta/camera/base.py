import abc

import numpy as np
from PyQt6 import QtCore, sip


class AbstractQObject(sip.wrappertype, abc.ABCMeta):
    pass


class ICamera(QtCore.QObject, abc.ABC, metaclass=AbstractQObject):

    acquisition_finished = QtCore.pyqtSignal(np.ndarray, np.ndarray, int, int)
    connection_closed = QtCore.pyqtSignal()
    connection_opened = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._number_of_scans = 100
        self._exposure_time_us = 1
        self._probe = np.zeros((self.number_of_scans, self.total_pixels))
        self._reference = np.zeros((self.number_of_scans, self.total_pixels))

    @abc.abstractmethod
    def __str__(self) -> str:
        """Display name of the camera hardware
        """

    @property
    def number_of_scans(self) -> int:
        return self._number_of_scans

    @number_of_scans.setter
    def number_of_scans(self, number_of_scans: int) -> None:
        self._number_of_scans = number_of_scans

    @property
    def exposure_time_us(self) -> int:
        return self._exposure_time_us

    @exposure_time_us.setter
    def exposure_time_us(self, exposure_time_us: int) -> None:
        self._exposure_time_us = exposure_time_us

    @property
    @abc.abstractmethod
    def total_pixels(self) -> int:
        """Total number of pixels in the wavelength direction of the camera array
        """

    @property
    @abc.abstractmethod
    def valid_pixels(self) -> int:
        """Number of pixels (out of ``total_pixels``) that contain valid data
        """

    @property
    @abc.abstractmethod
    def first_pixel(self) -> int:
        """First pixel in ``total_pixels`` that contains valid data
        """

    @property
    def probe(self) -> np.ndarray:
        return self._probe

    @property
    def reference(self) -> np.ndarray:
        return self._reference

    @QtCore.pyqtSlot()
    def initialise(self) -> None:
        self._connect()
        self.connection_opened.emit()

    @QtCore.pyqtSlot()
    def close(self) -> None:
        self._disconnect()
        self.connection_closed.emit()

    @QtCore.pyqtSlot()
    def acquire(self) -> None:
        self._read()
        self.acquisition_finished.emit(self.probe, self.reference, self.first_pixel, self.valid_pixels)
        self._overflow()

    @abc.abstractmethod
    def _read(self) -> None:
        """Update the ``self._probe`` and ``self._reference`` arrays by taking measurements
        """

    @abc.abstractmethod
    def _connect(self) -> None:
        """Connect to and initialise the camera(s)
        """

    @abc.abstractmethod
    def _disconnect(self) -> None:
        """Disconnect from the camera(s)
        """

    @abc.abstractmethod
    def _overflow(self) -> None:
        """If applicable, clear any overflow from the camera memory after reading
        """

    @abc.abstractmethod
    def update_number_of_scans(self, number_of_scans: int) -> None:
        """Call this to update the number of scans - the camera array may need to be re-initialised
        """
