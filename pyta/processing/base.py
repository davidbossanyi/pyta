import numpy as np

from pyta.processing.models import AcquisitionOnOffData, LinearPixelCorrection, ReferenceManipulationFactor


class AcquisitionProcessingMixin:
    @staticmethod
    def initialise_arrays(probe_array: np.ndarray, reference_array: np.ndarray, first_pixel: int, num_pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        untrimmed_probe_array = np.array(probe_array, dtype=int)
        raw_probe_array = np.array(probe_array, dtype=float)[:, first_pixel:num_pixels + first_pixel]
        raw_reference_array = np.array(reference_array, dtype=float)[:, first_pixel:num_pixels + first_pixel]
        return untrimmed_probe_array, raw_probe_array, raw_reference_array

    @staticmethod
    def separate_on_off(
        untrimmed_probe_array: np.ndarray,
        probe_array: np.ndarray,
        reference_array: np.ndarray,
        trigger_pixel: int,
        trigger_threshold: int,
        tau_flip_request: bool = False
    ) -> tuple[AcquisitionOnOffData, np.ndarray]:
        trigger_list = []
        for shot in untrimmed_probe_array:
            trigger_list.append(shot[trigger_pixel])
        trigger = np.array(trigger_list)
        if tau_flip_request is True:
            trigger = np.roll(trigger, 1)
        if (untrimmed_probe_array[0, trigger_pixel] >= trigger_threshold and not tau_flip_request) or (untrimmed_probe_array[0, trigger_pixel] < trigger_threshold and tau_flip_request):
            probe_on_array = probe_array[::2, :]
            probe_off_array = probe_array[1::2, :]
            reference_on_array = reference_array[::2, :]
            reference_off_array = reference_array[1::2, :]
        else:
            probe_on_array = probe_array[1::2, :]
            probe_off_array = probe_array[::2, :]
            reference_on_array = reference_array[1::2, :]
            reference_off_array = reference_array[::2, :]
        acquisition_on_off_data = AcquisitionOnOffData(
            probe_on=probe_on_array,
            probe_off=probe_off_array,
            reference_on=reference_on_array,
            reference_off=reference_off_array,
        )
        return acquisition_on_off_data, trigger

    @staticmethod
    def average_shots(acquisition_on_off_data: AcquisitionOnOffData) -> AcquisitionOnOffData:
        return AcquisitionOnOffData(
            probe_on=acquisition_on_off_data.probe_on.mean(axis=0),
            probe_off=acquisition_on_off_data.probe_off.mean(axis=0),
            reference_on=acquisition_on_off_data.reference_on.mean(axis=0),
            reference_off=acquisition_on_off_data.reference_off.mean(axis=0),
        )

    @staticmethod
    def subtract_bgd(acquisition: AcquisitionOnOffData, bgd: AcquisitionOnOffData) -> AcquisitionOnOffData:
        return AcquisitionOnOffData(
            probe_on=acquisition.probe_on - bgd.probe_on,
            probe_off=acquisition.probe_off - bgd.probe_off,
            reference_on=acquisition.reference_on - bgd.reference_on,
            reference_off=acquisition.reference_off - bgd.reference_off,
        )

    @staticmethod
    def set_linear_pixel_correlation(raw_probe_array: np.ndarray,
                                     raw_reference_array: np.ndarray) -> LinearPixelCorrection:
        pr_corr = raw_probe_array.mean(axis=0)
        ref_corr = raw_reference_array.mean(axis=0)
        pr_corr[::2] = pr_corr[::2] / pr_corr[1::2]
        ref_corr[::2] = ref_corr[::2] / ref_corr[1::2]
        pr_corr[1::2] = pr_corr[1::2] / pr_corr[1::2]
        ref_corr[1::2] = ref_corr[1::2] / ref_corr[1::2]
        linear_pixel_correction = LinearPixelCorrection(probe_correction=pr_corr, reference_correction=ref_corr)
        return linear_pixel_correction

    @staticmethod
    def linear_pixel_correction(probe_array: np.ndarray, reference_array: np.ndarray, lpc: LinearPixelCorrection) -> tuple[np.ndarray, np.ndarray]:
        probe_array = probe_array / lpc.probe_correction
        reference_array = reference_array / lpc.reference_correction
        return probe_array, reference_array

    @staticmethod
    def manipulate_reference(data: AcquisitionOnOffData, num_pixels: int, reference_manipulation_factors: ReferenceManipulationFactor) -> AcquisitionOnOffData:
        vs = reference_manipulation_factors.vertical_scale
        vo = reference_manipulation_factors.vertical_offset
        ho = reference_manipulation_factors.horizontal_offset
        sc = reference_manipulation_factors.scale_center
        sf = reference_manipulation_factors.scale_factor
        if vs <= 0:
            vs = 1
        if sf <= 0:
            sf = 1
        x = np.linspace(0, num_pixels - 1, num_pixels)
        new_x = ((x - sc) * sf) + sc - ho
        for i, spectra in enumerate(data.reference_off):
            data.reference_off[i] = np.interp(new_x, x, spectra * vs + vo)
        for i, spectra in enumerate(data.reference_on):
            data.reference_on[i] = np.interp(new_x, x, spectra * vs + vo)
        return data

    @staticmethod
    def correct_probe_with_reference(data: AcquisitionOnOffData) -> tuple[np.ndarray, np.ndarray]:
        refd_probe_on_array = data.probe_on / data.reference_on
        refd_probe_off_array = data.probe_off / data.reference_off
        return refd_probe_on_array, refd_probe_off_array

    @staticmethod
    def calculate_dtt(probe_on_array: np.ndarray, probe_off_array: np.ndarray, probe_off_avg: np.ndarray | None = None) -> np.ndarray:
        if probe_off_avg is not None:
            dtt_array = (probe_on_array - probe_off_array) / probe_off_avg
        else:
            dtt_array = (probe_on_array - probe_off_array) / probe_off_array
        dtt = dtt_array.mean(axis=0)
        return dtt

    @staticmethod
    def calculate_dtt_error(data: AcquisitionOnOffData, refd_probe_off_array: np.ndarray, data_avg: AcquisitionOnOffData | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if data_avg is not None:
            probe_shot_error = np.std(
                2 * (data.probe_on - data.probe_off) / (data_avg.probe_on + data_avg.probe_off), axis=0)
            ref_shot_error = np.std(
                2 * (data.reference_on - data.reference_off) / (data_avg.reference_on + data_avg.reference_off),
                axis=0)
        else:
            probe_shot_error = np.std(
                2 * (data.probe_on - data.probe_off) / (data.probe_on + data.probe_off),
                axis=0)
            ref_shot_error = np.std(2 * (data.reference_on - data.reference_off) / (
                        data.reference_on + data.reference_off), axis=0)
        dtt_error = np.std(refd_probe_off_array, axis=0)
        return probe_shot_error, ref_shot_error, dtt_error
