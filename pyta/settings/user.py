from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TimePointDistribution(Enum):
    LINEAR = 0
    EXPONENTIAL = 1


class UserSettings(BaseModel):
    use_linear_corr: bool = Field(False, alias="useLinearCorr")
    use_reference: bool = Field(True, alias="useReference")
    use_cutoff: bool = Field(True, alias="useCutoff")
    cutoff_pixel_low: int = Field(30, alias="cutoffPixelLow")
    cutoff_pixel_high: int = Field(1000, alias="cutoffPixelHigh")
    use_calibration: bool = Field(True, alias="useCalibration")
    calibration_pixel_low: int = Field(30, alias="calibrationPixelLow")
    calibration_pixel_high: int = Field(1000, alias="calibrationPixelHigh")
    calibration_wavelength_low: int = Field(1000, alias="calibrationWavelengthLow")
    calibration_wavelength_high: int = Field(1000, alias="calibrationWavelengthHigh")
    delay_time_zero: float = Field(0.0, alias="delayTimeZero")
    time_point_distribution: TimePointDistribution = Field(TimePointDistribution.LINEAR, alias="timePointDistribution")
    start_time: float = Field(-5.0, alias="startTime")
    end_time: float = Field(100.0, alias="endTime")
    num_points: int = Field(100, alias="numberOfPoints")
    num_shots: int = Field(100, alias="numberOfShots")
    num_sweeps: int = Field(3, alias="numberOfSweeps")
    use_reference_manipulation: bool = Field(False, alias="useReferenceManipulation")
    refman_horiz_offset: float = Field(0.0, alias="referenceManipulationHorizontalOffset")
    refman_scale_center: float = Field(1.0, alias="referenceManipulationScaleCenter")
    refman_scale_factor: float = Field(1.0, alias="referenceManipulationScaleFactor")
    refman_vertical_offset: float = Field(0.0, alias="referenceManipulationVerticalOffset")
    refman_vertical_stretch: float = Field(1.0, alias="referenceManipulationVerticalStretch")
    threshold_pixel: int = Field(0, alias="triggerPixel")
    threshold_value: int = Field(15000, alias="triggerThreshold")
    time: float = Field(0.0, alias="currentDelayTime")
    time_jog_step: float = Field(0.01, alias="timeJogStep")
    dc_shot_factor: float = Field(1.0, alias="darkCorrectionShotFactor")
