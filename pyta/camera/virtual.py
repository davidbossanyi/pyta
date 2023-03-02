from __future__ import annotations

import time

import numpy as np

from pyta.camera.base import ICamera


class VirtualCamera(ICamera):
    def __init__(self) -> None:
        self._pixels = 1200
        self._num_pixels = 1024
        self._first_pixel = 16
        super().__init__()

    def __str__(self) -> str:
        return "Virtual"

    @property
    def total_pixels(self) -> int:
        return self._pixels

    @property
    def valid_pixels(self) -> int:
        return self._num_pixels

    @property
    def first_pixel(self) -> int:
        return self._first_pixel

    def _connect(self) -> None:
        time.sleep(5)

    def _read(self) -> None:
        self._probe = np.random.normal(size=(self.number_of_scans, self.total_pixels))
        self._reference = np.random.normal(size=(self.number_of_scans, self.total_pixels))
        time.sleep(1)

    def _disconnect(self) -> None:
        time.sleep(1)

    def _overflow(self) -> None:
        pass

    def update_number_of_scans(self, number_of_scans: int) -> None:
        self.number_of_scans = number_of_scans
        self._probe = np.zeros((self.number_of_scans, self.total_pixels))
        self._reference = np.zeros((self.number_of_scans, self.total_pixels))
