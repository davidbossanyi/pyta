import dataclasses

import numpy as np


@dataclasses.dataclass
class AcquisitionData:
    probe_on: np.ndarray
    probe_on_array: np.ndarray
    probe_shot_error: np.ndarray
    reference_on: np.ndarray
    reference_on_array: np.ndarray
    ref_shot_error: np.ndarray
    dtt: np.ndarray
    dtt_error: np.ndarray
    trigger: np.ndarray
    high_trigger_std: bool
    high_dtt: bool


@dataclasses.dataclass
class AcquisitionOnOffData:
    probe_on: np.ndarray
    probe_off: np.ndarray
    reference_on: np.ndarray
    reference_off: np.ndarray


@dataclasses.dataclass
class LinearPixelCorrection:
    probe_correction: np.ndarray
    reference_correction: np.ndarray


@dataclasses.dataclass
class ReferenceManipulationFactor:
    vertical_scale: float
    vertical_offset: float
    horizontal_offset: float
    scale_center: float
    scale_factor: float
