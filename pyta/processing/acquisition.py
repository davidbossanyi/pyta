from __future__ import annotations

import numpy as np
from PyQt5 import QtCore

from pyta.processing.base import AcquisitionProcessingMixin
from pyta.processing.models import AcquisitionData, AcquisitionOnOffData, LinearPixelCorrection


class Acquisition(QtCore.QObject, AcquisitionProcessingMixin):
    processing_finished = QtCore.pyqtSignal()
    bgd_processing_finished = QtCore.pyqtSignal()

    def __init__(self, use_bgd_subtraction: bool = True, use_linear_pixel_correction: bool = False):
        super().__init__()
        self._use_bgd_subtraction = use_bgd_subtraction
        self._use_linear_pixel_correction = use_linear_pixel_correction
        self._background: AcquisitionOnOffData | None = None
        self._data: AcquisitionData | None = None
        self._linear_pixel_correction: LinearPixelCorrection | None = None

    @property
    def data(self) -> AcquisitionData | None:
        return self._data

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, int, int, int, int)
    def process_background(
        self,
        probe_array: np.ndarray,
        reference_array: np.ndarray,
        first_pixel: int,
        num_pixels: int,
        trigger_pixel: int,
        trigger_threshold: int,
    ) -> None:
        untrimmed_probe_array, raw_probe_array, raw_reference_array = self.initialise_arrays(
            probe_array, reference_array, first_pixel, num_pixels
        )
        probe_array = raw_probe_array.copy()
        reference_array = raw_reference_array.copy()
        if self._use_linear_pixel_correction:
            lpc = self._linear_pixel_correction = self.set_linear_pixel_correlation(raw_probe_array, raw_reference_array)
            probe_array, reference_array = self.linear_pixel_correction(probe_array, reference_array, lpc)
        bgd, _ = self.separate_on_off(untrimmed_probe_array, probe_array, reference_array, trigger_pixel, trigger_threshold)
        self._background = self.average_shots(bgd)
        self.bgd_processing_finished.emit()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, int, int, int, int, bool, bool, bool)
    def process_data(
        self,
        probe_array: np.ndarray,
        reference_array: np.ndarray,
        first_pixel: int,
        num_pixels: int,
        trigger_pixel: int,
        trigger_threshold: int,
        tflip: bool,
        use_reference: bool = True,
        use_average_off_shots: bool = True,
    ) -> None:
        reference_manipulation_factors = None  # ToDo: implement
        untrimmed_probe_array, raw_probe_array, raw_reference_array = self.initialise_arrays(
            probe_array, reference_array, first_pixel, num_pixels
        )
        probe_array = raw_probe_array.copy()
        reference_array = raw_reference_array.copy()
        if self._use_linear_pixel_correction:
            lpc = self._linear_pixel_correction = self.set_linear_pixel_correlation(raw_probe_array, raw_reference_array)
            probe_array, reference_array = self.linear_pixel_correction(probe_array, reference_array, lpc)
        data, trigger = self.separate_on_off(
            untrimmed_probe_array, probe_array, reference_array, trigger_pixel, trigger_threshold, tflip
        )
        if self._use_bgd_subtraction:
            data = self.subtract_bgd(data, self._background)
        if reference_manipulation_factors:
            data = self.manipulate_reference(data, num_pixels, reference_manipulation_factors)
        data_avg = self.average_shots(data)
        data_avg_for_error_calc = data_avg if use_average_off_shots else None
        if use_reference:
            refd_probe_on_array, refd_probe_off_array = self.correct_probe_with_reference(data)
            refd_probe_off = refd_probe_off_array.mean(axis=0)
            probe_off_avg = refd_probe_off if use_average_off_shots else None
            dtt = self.calculate_dtt(refd_probe_on_array, refd_probe_off_array, probe_off_avg)
            probe_shot_error, ref_shot_error, dtt_error = self.calculate_dtt_error(
                data, refd_probe_off_array, data_avg_for_error_calc
            )
        else:
            probe_off_avg = data_avg.probe_off if use_average_off_shots else None
            dtt = self.calculate_dtt(data.probe_on, data.probe_off, probe_off_avg)
            probe_shot_error, ref_shot_error, dtt_error = self.calculate_dtt_error(
                data, data.probe_off, data_avg_for_error_calc
            )
        self._data = AcquisitionData(
            probe_on=data_avg.probe_on,
            probe_on_array=data.probe_on,
            probe_shot_error=probe_shot_error,
            reference_on=data_avg.reference_on,
            reference_on_array=data.reference_on,
            ref_shot_error=ref_shot_error,
            dtt=dtt,
            dtt_error=dtt_error,
            trigger=trigger,
            high_dtt=False,  # ToDo: calculate
            high_trigger_std=False,
        )
        self.processing_finished.emit()
