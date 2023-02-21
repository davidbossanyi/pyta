from typing import Any

import numpy as np
import os
import h5py
import datetime as dt

from PyQt6 import QtCore


class Sweep(QtCore.QObject):

    data_saved = QtCore.pyqtSignal()

    def __init__(self, times: np.ndarray, num_pixels: int, filename: str, metadata: dict[str, Any]):
        super().__init__()
        self.hdf5_filename = self.generate_unique_filename(filename)
        self.metadata = metadata
        self._sweep_index = 0
        self.times = np.array(times, ndmin=2)
        self._sweep_index_array = np.zeros(shape=(self.times.size, 1))
        self._pixels = np.linspace(0, num_pixels - 1, num_pixels)
        self._current_data = np.zeros(shape=(self.times.size, num_pixels))
        self._avg_data = np.zeros(shape=(self.times.size, num_pixels))

    @property
    def sweep_index(self) -> int:
        return self._sweep_index

    @property
    def avg_data(self) -> np.ndarray:
        return self._avg_data

    @property
    def current_data(self) -> np.ndarray:
        return self._current_data

    @staticmethod
    def generate_unique_filename(filename: str) -> str:
        filename = os.path.splitext(filename)[0]
        hdf5_filename = f"{filename}.hdf5"
        i = 1
        while os.path.isfile(hdf5_filename):
            hdf5_filename = f"{filename}_{i:03}.hdf5"
            i += 1
        return hdf5_filename

    def add_current_data(self, dtt: np.ndarray, time_point: int) -> None:
        self._current_data[time_point, :] = dtt
        if self.sweep_index == 0:
            self._avg_data[time_point, :] = dtt
        else:
            self._avg_data[time_point, :] = np.array(
                ((self._avg_data[time_point, :] * self._sweep_index_array[time_point]) + dtt) / (
                            self._sweep_index_array[time_point] + 1))
        self._sweep_index_array[time_point] = self._sweep_index_array[time_point] + 1

    def next_sweep(self) -> None:
        self._sweep_index += 1
        self._current_data = np.zeros(shape=(self.times.size, self._pixels.size))

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def save(self, wavelengths: np.ndarray, probe: np.ndarray, reference: np.ndarray, error: np.ndarray) -> None:
        self._save_current_data(wavelengths)
        self._save_avg_data(wavelengths)
        self._save_metadata_each_sweep(probe, reference, error)
        self.data_saved.emit()

    def _save_current_data(self, waves: np.ndarray) -> None:
        save_data = np.vstack((np.hstack((0, waves)),
                               np.hstack((self.times.T,
                                          self.current_data))))

        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            dset = hdf5_file.create_dataset('Sweeps/Sweep_' + str(self.sweep_index), data=save_data)
            dset.attrs['date'] = str(dt.datetime.now().date()).encode('ascii', 'ignore')
            dset.attrs['time'] = str(dt.datetime.now().time()).encode('ascii', 'ignore')

    def _save_avg_data(self, waves: np.ndarray) -> None:
        save_data = np.vstack((np.hstack((0, waves)),
                               np.hstack((self.times.T,
                                          self.avg_data))))

        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            if "Average" in hdf5_file:
                dset = hdf5_file['Average']
                dset[:, :] = save_data
                dset.attrs.modify('end_date', str(dt.datetime.now().date()).encode('ascii', 'ignore'))
                dset.attrs.modify('end_time', str(dt.datetime.now().time()).encode('ascii', 'ignore'))
                dset.attrs.modify('num_sweeps', str(self.sweep_index).encode('ascii', 'ignore'))
            else:
                self._save_metadata_initial()
                dset = hdf5_file.create_dataset('Average', data=save_data)
                dset.attrs['start date'] = str(dt.datetime.now().date()).encode('ascii', 'ignore')
                dset.attrs['start time'] = str(dt.datetime.now().time()).encode('ascii', 'ignore')
                for key, item in self.metadata.items():
                    dset.attrs[key] = str(item).encode('ascii', 'ignore')
                dset.attrs['num_sweeps'] = str(self.sweep_index).encode('ascii', 'ignore')

    def _save_metadata_initial(self) -> None:
        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            data = np.zeros((1, 1))
            dset = hdf5_file.create_dataset('Metadata', data=data)
            for key, item in self.metadata.items():
                dset.attrs[key] = str(item).encode('ascii', 'ignore')

    def _save_metadata_each_sweep(self, probe: np.ndarray, reference: np.ndarray, error: np.ndarray) -> None:
        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            dset = hdf5_file.create_dataset(f'Spectra/Sweep_{self.sweep_index}_Probe_Spectrum', data=probe)
            dset.attrs['date'] = str(dt.datetime.now().date()).encode('ascii', 'ignore')
            dset.attrs['time'] = str(dt.datetime.now().time()).encode('ascii', 'ignore')

            dset2 = hdf5_file.create_dataset(f'Spectra/Sweep_{self.sweep_index}_Reference_Spectrum',
                                             data=reference)
            dset2.attrs['date'] = str(dt.datetime.now().date()).encode('ascii', 'ignore')
            dset2.attrs['time'] = str(dt.datetime.now().time()).encode('ascii', 'ignore')

            dset3 = hdf5_file.create_dataset(f'Spectra/Sweep_{self.sweep_index}_Error_Spectrum', data=error)
            dset3.attrs['date'] = str(dt.datetime.now().date()).encode('ascii', 'ignore')
            dset3.attrs['time'] = str(dt.datetime.now().time()).encode('ascii', 'ignore')
