from __future__ import annotations

import time

import numpy as np

from pyta.delay.base import IDelay


class VirtualDelay(IDelay):

    def __str__(self) -> str:
        return "Virtual"

    @property
    def tmax(self) -> float:
        return self._time_zero + 300.0

    @property
    def tmin(self) -> float:
        return self._time_zero - 50.0

    def _connect(self) -> None:
        time.sleep(5)

    def _disconnect(self) -> None:
        time.sleep(1)

    def _move(self, time_point_ps: float) -> bool:
        time.sleep(1)
        return False

    def home(self) -> None:
        time.sleep(5)

    def check_times(self, time_points: np.ndarray) -> bool:
        return all((time_point <= self.tmax) & (time_point >= self.tmin) for time_point in time_points)

    def check_time(self, time_point: float) -> bool:
        return (time_point <= self.tmax) & (time_point >= self.tmin)
