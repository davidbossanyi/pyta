from __future__ import annotations

import sys
import os

import pandas as pd
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot

from pyta.processing.models import AcquisitionData
from pyta.ui.gui import Ui_pyTAgui as pyTAgui
import pyqtgraph as pg
import numpy as np
from pyta.processing.acquisition import Acquisition
from pyta.processing.sweep import Sweep
from pyta.camera.virtual import VirtualCamera as Camera
from pyta.camera.base import ICamera
from pyta.delay.virtual import VirtualDelay as Delay
from pyta.delay.base import IDelay


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Application(QtWidgets.QMainWindow):

    camera_connection_requested = QtCore.pyqtSignal()
    camera_disconnection_requested = QtCore.pyqtSignal()

    delay_connection_requested = QtCore.pyqtSignal()
    delay_disconnection_requested = QtCore.pyqtSignal()

    acquisition_background_processing_requested = QtCore.pyqtSignal(np.ndarray, np.ndarray, int, int, int, int)

    acquisition_requested = QtCore.pyqtSignal()
    acquisition_processing_requested = QtCore.pyqtSignal(np.ndarray, np.ndarray, int, int, int, int, bool, bool, bool)

    save_sweep_requested = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    move_delay_requested = QtCore.pyqtSignal(float)

    def __init__(self, last_instance_filename: str, last_instance_values: pd.DataFrame | None = None, preloaded: bool = False):
        super().__init__()
        self.ui = pyTAgui()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('assets/icon.png'))
        self.ui.tabs.setCurrentIndex(0)
        self.ui.diagnostics_tab.setEnabled(False)
        self.ui.acquisition_tab.setEnabled(False)
        self.last_instance_filename = last_instance_filename
        self.last_instance_values = last_instance_values
        self.preloaded = preloaded

        self.camera_connected = False
        self.camera: ICamera = Camera()
        self.camera_thread = QtCore.QThread()
        self.camera.moveToThread(self.camera_thread)
        self.camera.connection_opened.connect(self.post_camera_initialised)
        self.camera_connection_requested.connect(self.camera.initialise)
        self.camera_disconnection_requested.connect(self.camera.close)
        self.camera.connection_closed.connect(self.post_camera_disconnected)
        self.acquisition_requested.connect(self.camera.acquire)

        self.use_ir_gain = False
        self.use_logscale = False

        self.filename = ""
        self.filepath = os.path.expanduser("~")

        self.delay_connected = False
        self.delay: IDelay = Delay(0.0)
        self.delay_thread = QtCore.QThread()
        self.delay.moveToThread(self.delay_thread)
        self.delay.connection_opened.connect(self.post_delay_initialised)
        self.delay_connection_requested.connect(self.delay.initialise)
        self.delay_disconnection_requested.connect(self.delay.close)
        self.delay.connection_closed.connect(self.post_delay_disconnected)
        self.move_delay_requested.connect(self.delay.move_to)

        self.timeunits = 'ps'
        self.xlabel = 'Wavelength / Pixel'
        self.datafolder = os.path.join(os.path.expanduser('~'), 'Documents')
        self.timefile_folder = os.path.join(os.path.expanduser('~'), 'Documents')
        self.initialize_gui_values()
        self.setup_gui_connections()
        self.metadata = {}
        self.idle = True
        self.finished_acquisition = False
        self.safe_to_exit = True
        self.initialise_gui()
        self.show()

        self.use_timefile = self.ui.a_timefile_cb.isChecked()
        self.times = np.array([0.0])
        self.tau_flip_request = False

        self.probe_error_region = pg.FillBetweenItem(brush=(255, 0, 0, 50))
        self.ref_error_region = pg.FillBetweenItem(brush=(0, 0, 255, 50))

        self.plot_waves = np.linspace(0, self.camera.valid_pixels - 1, self.camera.valid_pixels)
        self.plot_times = self.times

        self.acquisition = Acquisition()
        self.processing_thread = QtCore.QThread()
        self.acquisition.moveToThread(self.processing_thread)
        self.acquisition_processing_requested.connect(self.acquisition.process_data)
        self.acquisition_background_processing_requested.connect(self.acquisition.process_background)
        self.current_data: AcquisitionData | None = None
        self.current_sweep: Sweep | None = None
        self.timestep = 0

        self.save_thread = QtCore.QThread()

        self.stop_request = False
        self.diagnostics_on = False

        self.write_app_status('application launched', colour='blue')
        
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.safe_to_exit:
            self.camera_thread.deleteLater()
            self.camera.deleteLater()
            self.delay_thread.deleteLater()
            self.delay.deleteLater()
            self.processing_thread.deleteLater()
            self.save_thread.deleteLater()
            self.save_gui_values()
            event.accept()
        else:
            event.ignore()
            self.message_box(text="Unable to quit safely. Please disconnect all hardware and try again.")
            
    def write_app_status(self, message: str, colour: str, timeout: int = 0) -> None:
        self.ui.statusBar.clearMessage()
        self.ui.statusBar.setStyleSheet('QStatusBar{color:'+colour+';}')
        self.ui.statusBar.showMessage(message, msecs=timeout)

    def initialize_gui_values(self) -> None:
        # dropdown menus
        self.ui.a_delaytype_dd.addItem(str(self.delay))
        self.ui.a_delaytype_dd.setEnabled(False)
        self.ui.d_delaytype_dd.addItem(str(self.delay))
        self.ui.d_delaytype_dd.setEnabled(False)
        self.ui.d_use_ir_gain.setEnabled(False)
        self.ui.d_display_mode_spectra.addItem('Probe')
        self.ui.d_display_mode_spectra.addItem('Reference')
        self.ui.a_distribution_dd.addItem('Exponential')
        self.ui.a_distribution_dd.addItem('Linear')
        self.ui.h_camera_dd.addItem(str(self.camera))
        self.ui.h_delay_dd.addItem(str(self.delay))
        # progress bars
        self.ui.a_measurement_progress_bar.setValue(0)
        self.ui.a_sweep_progress_bar.setValue(0)
        # other stuff
        self.ui.a_filename_le.setText('newfile')
        self.ui.a_use_calib.toggle()
        self.ui.a_use_cutoff.toggle()
        self.ui.a_plot_log_t_cb.setChecked(False)
        self.ui.a_plot_log_t_cb.setEnabled(False)

        self.ui.d_use_linear_corr.setChecked(False)
        self.ui.d_use_reference.setChecked(True)
        # file
        if self.preloaded is False:
            self.ui.a_use_cutoff.setChecked(True)
            self.ui.a_cutoff_pixel_low.setValue(30)
            self.ui.a_cutoff_pixel_high.setValue(1000)
            self.ui.d_cutoff_pixel_low.setValue(30)
            self.ui.d_cutoff_pixel_high.setValue(1000)
            self.ui.a_use_calib.setChecked(True)
            self.ui.a_calib_pixel_low.setValue(200)
            self.ui.a_calib_pixel_high.setValue(800)
            self.ui.a_calib_wave_low.setValue(400)
            self.ui.a_calib_wave_high.setValue(700)
            self.ui.d_calib_pixel_low.setValue(200)
            self.ui.d_calib_pixel_high.setValue(800)
            self.ui.d_calib_wave_low.setValue(400)
            self.ui.d_calib_wave_high.setValue(700)
            self.ui.a_delay_t0.setValue(0)
            self.ui.d_delay_t0.setValue(0)
            self.ui.a_delaytype_dd.setCurrentIndex(0)
            self.ui.a_distribution_dd.setCurrentIndex(0)
            self.ui.a_tstart_sb.setValue(-5)
            self.ui.a_tend_sb.setValue(100)
            self.ui.a_num_tpoints_sb.setValue(100)
            self.ui.a_num_shots.setValue(200)
            self.ui.a_num_sweeps.setValue(500)
            self.ui.d_display_mode_spectra.setCurrentIndex(0)
            self.ui.d_use_ref_manip.setChecked(True)
            self.ui.d_refman_horiz_offset.setValue(0)
            self.ui.d_refman_scale_center.setValue(250)
            self.ui.d_refman_scale_factor.setValue(1)
            self.ui.d_refman_vertical_offset.setValue(0)
            self.ui.d_refman_vertical_stretch.setValue(1)
            self.ui.d_threshold_pixel.setValue(0)
            self.ui.d_threshold_value.setValue(15000)
            self.ui.d_time.setValue(100)
            self.ui.d_jogstep_sb.setValue(0.01)
            self.ui.d_use_linear_corr.setChecked(False)
        else:
            self.ui.a_use_cutoff.setChecked(self.last_instance_values['use cutoff'])
            self.ui.a_cutoff_pixel_low.setValue(self.last_instance_values['cutoff pixel low'])
            self.ui.a_cutoff_pixel_high.setValue(self.last_instance_values['cutoff pixel high'])
            self.ui.d_cutoff_pixel_low.setValue(self.last_instance_values['cutoff pixel low'])
            self.ui.d_cutoff_pixel_high.setValue(self.last_instance_values['cutoff pixel high'])
            self.ui.a_use_calib.setChecked(self.last_instance_values['use calib'])
            self.ui.a_calib_pixel_low.setValue(self.last_instance_values['calib pixel low'])
            self.ui.a_calib_pixel_high.setValue(self.last_instance_values['calib pixel high'])
            self.ui.a_calib_wave_low.setValue(self.last_instance_values['calib wavelength low'])
            self.ui.a_calib_wave_high.setValue(self.last_instance_values['calib wavelength high'])
            self.ui.d_calib_pixel_low.setValue(self.last_instance_values['calib pixel low'])
            self.ui.d_calib_pixel_high.setValue(self.last_instance_values['calib pixel high'])
            self.ui.d_calib_wave_low.setValue(self.last_instance_values['calib wavelength low'])
            self.ui.d_calib_wave_high.setValue(self.last_instance_values['calib wavelength high'])
            self.ui.a_delay_t0.setValue(self.last_instance_values['short stage time zero'])
            self.ui.d_delay_t0.setValue(self.last_instance_values['long stage time zero'])
            self.ui.a_delaytype_dd.setCurrentIndex(self.last_instance_values['delay type'])
            self.ui.a_distribution_dd.setCurrentIndex(self.last_instance_values['distribution'])
            self.ui.a_tstart_sb.setValue(self.last_instance_values['tstart'])
            self.ui.a_tend_sb.setValue(self.last_instance_values['tend'])
            self.ui.a_num_tpoints_sb.setValue(self.last_instance_values['num tpoints'])
            self.ui.a_num_shots.setValue(self.last_instance_values['num shots'])
            self.ui.a_num_sweeps.setValue(self.last_instance_values['num sweeps'])
            self.ui.d_display_mode_spectra.setCurrentIndex(self.last_instance_values['d display mode spectra'])
            self.ui.d_use_ref_manip.setChecked(self.last_instance_values['d use ref manip'])
            self.ui.d_refman_horiz_offset.setValue(self.last_instance_values['d refman horizontal offset'])
            self.ui.d_refman_scale_center.setValue(self.last_instance_values['d refman scale center'])
            self.ui.d_refman_scale_factor.setValue(self.last_instance_values['d refman scale factor'])
            self.ui.d_refman_vertical_offset.setValue(self.last_instance_values['d refman vertical offset'])
            self.ui.d_refman_vertical_stretch.setValue(self.last_instance_values['d refman vertical stretch'])
            self.ui.d_threshold_pixel.setValue(self.last_instance_values['d threshold pixel'])
            self.ui.d_threshold_value.setValue(self.last_instance_values['d threshold value'])
            self.ui.d_time.setValue(self.last_instance_values['d time'])
            self.ui.d_jogstep_sb.setValue(self.last_instance_values['d jogstep'])
        
    def setup_gui_connections(self) -> None:
        # acquisition file stuff
        self.ui.a_folder_btn.clicked.connect(self.exec_folder_browse_btn)
        self.ui.a_filename_le.textChanged.connect(self.update_filepath)
        self.ui.a_metadata_pump_wavelength.textChanged.connect(self.metadata_changed)
        self.ui.a_metadata_pump_power.textChanged.connect(self.metadata_changed)
        self.ui.a_metadata_pump_spotsize.textChanged.connect(self.metadata_changed)
        self.ui.a_metadata_probe_wavelength.textChanged.connect(self.metadata_changed)
        self.ui.a_metadata_probe_power.textChanged.connect(self.metadata_changed)
        self.ui.a_metadata_probe_spotsize.textChanged.connect(self.metadata_changed)
        # acquisition times options
        self.ui.a_distribution_dd.currentIndexChanged.connect(self.update_times)
        self.ui.a_tstart_sb.valueChanged.connect(self.update_times)
        self.ui.a_tend_sb.valueChanged.connect(self.update_times)
        self.ui.a_num_tpoints_sb.valueChanged.connect(self.update_times)
        self.ui.a_timefile_cb.toggled.connect(self.update_use_timefile)
        self.ui.a_timefile_btn.clicked.connect(self.exec_timefile_folder_btn)
        self.ui.a_timefile_list.currentIndexChanged.connect(self.update_times_from_file)
        # aquisition acquire options
        self.ui.a_delay_t0.valueChanged.connect(self.update_delay_t0)
        self.ui.a_num_shots.valueChanged.connect(self.update_num_shots)
        self.ui.a_num_sweeps.valueChanged.connect(self.update_num_sweeps)
        # acquisition calibration
        self.ui.a_use_calib.toggled.connect(self.update_use_calib)
        self.ui.a_calib_pixel_low.valueChanged.connect(self.update_calib)
        self.ui.a_calib_pixel_high.valueChanged.connect(self.update_calib)
        self.ui.a_calib_wave_low.valueChanged.connect(self.update_calib)
        self.ui.a_calib_wave_high.valueChanged.connect(self.update_calib)
        # acquisition cutoff
        self.ui.a_use_cutoff.toggled.connect(self.update_use_cutoff)
        self.ui.a_cutoff_pixel_low.valueChanged.connect(self.update_cutoff)
        self.ui.a_cutoff_pixel_high.valueChanged.connect(self.update_cutoff)
        # acquisition launch
        self.ui.a_run_btn.clicked.connect(self.exec_run_btn)
        self.ui.a_stop_btn.clicked.connect(self.exec_stop_btn)
        # acquisition plot options
        self.ui.a_plot_log_t_cb.toggled.connect(self.update_plot_log_t)
        # diagnostics reference manipulation
        self.ui.d_refman_vertical_stretch.valueChanged.connect(self.update_refman)
        self.ui.d_refman_vertical_offset.valueChanged.connect(self.update_refman)
        self.ui.d_refman_horiz_offset.valueChanged.connect(self.update_refman)
        self.ui.d_refman_scale_center.valueChanged.connect(self.update_refman)
        self.ui.d_refman_scale_factor.valueChanged.connect(self.update_refman)
        # diagnostics calibration
        self.ui.d_use_calib.toggled.connect(self.update_d_use_calib)
        self.ui.d_calib_pixel_low.valueChanged.connect(self.update_d_calib)
        self.ui.d_calib_pixel_high.valueChanged.connect(self.update_d_calib)
        self.ui.d_calib_wave_low.valueChanged.connect(self.update_d_calib)
        self.ui.d_calib_wave_high.valueChanged.connect(self.update_d_calib)
        # diagnostics cutoff
        self.ui.d_use_cutoff.toggled.connect(self.update_d_use_cutoff)
        self.ui.d_cutoff_pixel_low.valueChanged.connect(self.update_d_cutoff)
        self.ui.d_cutoff_pixel_high.valueChanged.connect(self.update_d_cutoff)
        # diagnstics aquire options
        self.ui.d_delay_t0.valueChanged.connect(self.update_d_delay_t0)
        self.ui.d_num_shots.valueChanged.connect(self.update_d_num_shots)
        self.ui.d_dcshotfactor_sb.valueChanged.connect(self.update_d_dcshotfactor)
        # diagnostics time
        self.ui.d_time.valueChanged.connect(self.update_d_time)
        self.ui.d_move_to_time_btn.clicked.connect(self.exec_d_move_to_time)
        self.ui.d_jogstep_sb.valueChanged.connect(self.update_d_jogstep)
        self.ui.d_jogleft.clicked.connect(self.d_jog_earlier)
        self.ui.d_jogright.clicked.connect(self.d_jog_later)
        self.ui.d_set_current_btn.clicked.connect(self.exec_d_set_current_btn)
        # diagnostics other
        self.ui.d_threshold_pixel.valueChanged.connect(self.update_threshold)
        self.ui.d_threshold_value.valueChanged.connect(self.update_threshold)
        # diagnostics launch
        self.ui.d_run_btn.clicked.connect(self.exec_d_run_btn)
        self.ui.d_stop_btn.clicked.connect(self.exec_d_stop_btn)
        # hardware camera
        self.ui.h_connect_camera_btn.clicked.connect(self.exec_h_camera_connect_btn)
        self.ui.h_disconnect_camera_btn.clicked.connect(self.exec_h_camera_disconnect_btn)
        # hardware delays
        self.ui.h_connect_delay_btn.clicked.connect(self.exec_h_delay_connect_btn)
        self.ui.h_disconnect_delay_btn.clicked.connect(self.exec_h_delay_disconnect_btn)
        
    def initialise_gui(self) -> None:
        self.update_delay_t0()
        self.update_calib()
        self.update_cutoff()
        self.update_num_shots()
        self.update_num_sweeps()
        self.update_plot_log_t()
        self.update_refman()
        self.update_threshold()
        self.update_use_calib()
        self.update_use_cutoff()
        self.update_d_time()
        self.update_d_jogstep()
        self.update_use_timefile()
        self.update_use_ir_gain()
        self.update_times()
        self.update_xlabel()
        self.update_filepath()
        self.update_d_dcshotfactor()
        
    def save_gui_values(self) -> None:
        self.last_instance_values['use cutoff'] = self.ui.a_use_cutoff.isChecked()
        self.last_instance_values['cutoff pixel low'] = self.ui.a_cutoff_pixel_low.value()
        self.last_instance_values['cutoff pixel high'] = self.ui.a_cutoff_pixel_high.value()
        self.last_instance_values['use calib'] = self.ui.a_use_calib.isChecked()
        self.last_instance_values['calib pixel low'] = self.ui.a_calib_pixel_low.value()
        self.last_instance_values['calib pixel high'] = self.ui.a_calib_pixel_high.value()
        self.last_instance_values['calib wavelength low'] = self.ui.a_calib_wave_low.value()
        self.last_instance_values['calib wavelength high'] = self.ui.a_calib_wave_high.value()
        self.last_instance_values['short stage time zero'] = self.ui.a_delay_t0.value()
        self.last_instance_values['delay type'] = self.ui.a_delaytype_dd.currentIndex()
        self.last_instance_values['distribution'] = self.ui.a_distribution_dd.currentIndex()
        self.last_instance_values['tstart'] = self.ui.a_tstart_sb.value()
        self.last_instance_values['tend'] = self.ui.a_tend_sb.value()
        self.last_instance_values['num tpoints'] = self.ui.a_num_tpoints_sb.value()
        self.last_instance_values['num shots'] = self.ui.a_num_shots.value()
        self.last_instance_values['num sweeps'] = self.ui.a_num_sweeps.value()
        self.last_instance_values['d display mode spectra'] = self.ui.d_display_mode_spectra.currentIndex()
        self.last_instance_values['d use ref manip'] = self.ui.d_use_ref_manip.isChecked()
        self.last_instance_values['d refman horizontal offset'] = self.ui.d_refman_horiz_offset.value()
        self.last_instance_values['d refman scale center'] = self.ui.d_refman_scale_center.value()
        self.last_instance_values['d refman scale factor'] = self.ui.d_refman_scale_factor.value()
        self.last_instance_values['d refman vertical offset'] = self.ui.d_refman_vertical_offset.value()
        self.last_instance_values['d refman vertical stretch'] = self.ui.d_refman_vertical_stretch.value()
        self.last_instance_values['d threshold pixel'] = self.ui.d_threshold_pixel.value()
        self.last_instance_values['d threshold value'] = self.ui.d_threshold_value.value()
        self.last_instance_values['d time'] = self.ui.d_time.value()
        self.last_instance_values['d jogstep'] = self.ui.d_jogstep_sb.value()
        self.last_instance_values.to_csv(self.last_instance_filename, sep=':', header=False)

    @QtCore.pyqtSlot()
    def exec_h_camera_connect_btn(self) -> None:
        self.ui.h_connect_camera_btn.setEnabled(False)
        self.ui.h_camera_dd.setEnabled(False)
        self.h_update_camera_status('initialising... please wait')
        self.camera_thread.start()
        self.camera_connection_requested.emit()

    @QtCore.pyqtSlot()
    def post_camera_initialised(self) -> None:
        self.h_update_camera_status('ready')
        self.ui.h_disconnect_camera_btn.setEnabled(True)
        self.camera_connected = True
        self.safe_to_exit = False
        if self.delay_connected:
            self.ui.acquisition_tab.setEnabled(True)
            self.ui.diagnostics_tab.setEnabled(True)

    @QtCore.pyqtSlot()
    def exec_h_camera_disconnect_btn(self) -> None:
        self.ui.h_disconnect_camera_btn.setEnabled(False)
        self.camera_disconnection_requested.emit()

    @QtCore.pyqtSlot()
    def post_camera_disconnected(self) -> None:
        self.h_update_camera_status('ready to connect')
        self.ui.h_connect_camera_btn.setEnabled(True)
        self.ui.h_camera_dd.setEnabled(True)
        self.camera_connected = False
        if not self.delay_connected:
            self.safe_to_exit = True
        self.ui.acquisition_tab.setEnabled(False)
        self.ui.diagnostics_tab.setEnabled(False)
        self.camera_thread.quit()

    @QtCore.pyqtSlot()
    def h_update_camera_status(self, message: str) -> None:
        self.ui.h_camera_status.setText(message)

    @QtCore.pyqtSlot()
    def update_use_ir_gain(self) -> None:
        self.use_ir_gain = self.ui.h_use_ir_gain.isChecked()
        self.ui.d_use_ir_gain.setChecked(self.use_ir_gain)

    @QtCore.pyqtSlot()
    def exec_h_delay_connect_btn(self) -> None:
        self.ui.h_connect_delay_btn.setEnabled(False)
        self.ui.h_delay_dd.setEnabled(False)
        self.h_update_delay_status('initialising... please wait')
        self.delay_thread.start()
        self.delay_connection_requested.emit()

    @QtCore.pyqtSlot()
    def post_delay_initialised(self) -> None:
        self.h_update_delay_status('ready')
        self.ui.h_disconnect_delay_btn.setEnabled(True)
        self.delay_connected = True
        self.safe_to_exit = False
        if self.camera_connected:
            self.ui.acquisition_tab.setEnabled(True)
            self.ui.diagnostics_tab.setEnabled(True)

    @QtCore.pyqtSlot()
    def exec_h_delay_disconnect_btn(self) -> None:
        self.ui.h_disconnect_delay_btn.setEnabled(False)
        self.delay_disconnection_requested.emit()

    @QtCore.pyqtSlot()
    def post_delay_disconnected(self) -> None:
        self.h_update_delay_status('ready to connect')
        self.ui.h_connect_delay_btn.setEnabled(True)
        self.ui.h_delay_dd.setEnabled(True)
        self.delay_connected = False
        if not self.camera_connected:
            self.safe_to_exit = True
        self.ui.acquisition_tab.setEnabled(False)
        self.ui.diagnostics_tab.setEnabled(False)

    @QtCore.pyqtSlot()
    def h_update_delay_status(self, message: str) -> None:
        self.ui.h_delay_status.setText(message)

    @QtCore.pyqtSlot()
    def update_filepath(self) -> None:
        self.filename = self.ui.a_filename_le.text()
        self.filepath = os.path.join(self.datafolder, self.filename)
        self.ui.a_filepath_le.setText(self.filepath)

    @QtCore.pyqtSlot()
    def exec_folder_browse_btn(self) -> None:
        self.datafolder = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Folder', self.datafolder)
        self.datafolder = os.path.normpath(self.datafolder)
        self.update_filepath()

    def metadata_changed(self) -> None:
        self.metadata['pump wavelength'] = self.ui.a_metadata_pump_wavelength.text()
        self.metadata['pump power'] = self.ui.a_metadata_pump_power.text()
        self.metadata['pump size'] = self.ui.a_metadata_pump_spotsize.text()
        self.metadata['probe wavelengths'] = self.ui.a_metadata_probe_wavelength.text()
        self.metadata['probe power'] = self.ui.a_metadata_probe_power.text()
        self.metadata['probe size'] = self.ui.a_metadata_probe_power.text()
    
    def update_metadata(self) -> None:
        self.metadata_changed()
        self.metadata['delay type'] = str(self.delay)
        self.metadata['time zero'] = self.delay.time_zero
        self.metadata['num shots'] = self.num_shots
        self.metadata['calib pixel low'] = self.calib[0]
        self.metadata['calib pixel high'] = self.calib[1]
        self.metadata['calib wave low'] = self.calib[2]
        self.metadata['calib wave high'] = self.calib[3]
        self.metadata['cutoff low'] = self.cutoff[0]
        self.metadata['cutoff high'] = self.cutoff[1]
        self.metadata['use reference'] = self.ui.d_use_reference.isChecked()
        self.metadata['avg off shots'] = self.ui.d_use_avg_off_shots.isChecked()
        self.metadata['use ref manip'] = self.ui.d_use_ref_manip.isChecked()
        self.metadata['use calib'] = self.ui.d_use_calib.isChecked()
        
    def update_use_timefile(self) -> None:
        self.use_timefile = self.ui.a_timefile_cb.isChecked()
        if self.use_timefile:
            self.ui.a_distribution_dd.setEnabled(False)
            self.ui.a_tstart_sb.setEnabled(False)
            self.ui.a_tend_sb.setEnabled(False)
            self.ui.a_num_tpoints_sb.setEnabled(False)
            self.ui.a_timefile_btn.setEnabled(True)
            self.ui.a_timefile_list.setEnabled(True)
        else:
            self.ui.a_distribution_dd.setEnabled(True)
            self.ui.a_tstart_sb.setEnabled(True)
            self.ui.a_tend_sb.setEnabled(True)
            self.ui.a_num_tpoints_sb.setEnabled(True)
            self.ui.a_timefile_btn.setEnabled(False)
            self.ui.a_timefile_list.setEnabled(False)
            self.update_times()

    def update_times(self) -> None:
        distribution = self.ui.a_distribution_dd.currentText()
        if distribution == 'Linear':
            self.ui.a_num_tpoints_sb.setMinimum(5)
        else:
            self.ui.a_num_tpoints_sb.setMinimum(25)
        start_time = self.ui.a_tstart_sb.value()
        end_time = self.ui.a_tend_sb.value()
        num_points = self.ui.a_num_tpoints_sb.value()
        times = np.linspace(start_time, end_time, num_points)
        if distribution == 'Exponential':
            times = self.calculate_times_exponential(start_time, end_time, num_points)
        self.times = times
        self.display_times()
    
    def exec_timefile_folder_btn(self) -> None:
        self.timefile = QtWidgets.QFileDialog.getOpenFileName(None, 'Select TimeFile', self.timefile_folder, 'TimeFiles (*.tf)')
        self.timefile_folder = os.path.dirname(self.timefile[0])
        if self.timefile_folder.endswith('/'):
            self.timefile_folder = self.timefile_folder[:-1]
        self.timefile = os.path.basename(self.timefile[0])
        self.load_timefiles_to_list()

    def load_timefiles_to_list(self) -> None:
        self.ui.a_timefile_list.clear()
        self.timefiles: list[str] = []
        for file in os.listdir(self.timefile_folder):
            if file.endswith('.tf'):
                self.timefiles.append(file)
        current_index = 0
        for i, timefile in enumerate(self.timefiles):
            self.ui.a_timefile_list.addItem(timefile)
            if timefile == self.timefile:
                current_index = i
        self.ui.a_timefile_list.setCurrentIndex(current_index)
        self.update_times_from_file()

    def update_times_from_file(self) -> None:
        self.timefile = self.timefiles[self.ui.a_timefile_list.currentIndex()]
        self.times = np.genfromtxt(os.path.join(self.timefile_folder, self.timefile), dtype=float)
        self.display_times()

    def display_times(self) -> None:
        self.ui.a_times_list.clear()
        for time in self.times:
            self.ui.a_times_list.appendPlainText('{0:.2f}'.format(time))

    @staticmethod
    def calculate_times_exponential(start_time: float, end_time: float, num_points: int) -> np.ndarray:
        num_before_zero = 20
        step = 0.1
        before_zero = np.linspace(start_time, 0, num_before_zero, endpoint=False)
        zero_onwards = np.geomspace(step, end_time+step, num_points-num_before_zero)-step
        times = np.concatenate((before_zero, zero_onwards))
        return times
    
    def update_d_time_box_limits(self) -> None:
        self.ui.d_time.setMaximum(self.delay.tmax)
        self.ui.d_time.setMinimum(self.delay.tmin)

    def update_delay_t0(self) -> None:
        self.delay.time_zero = self.ui.a_delay_t0.value()
        self.ui.d_delay_t0.setValue(self.delay.time_zero)
        self.update_d_time_box_limits()
        
    def update_d_delay_t0(self) -> None:
        self.delay.time_zero = self.ui.d_delay_t0.value()
        self.ui.a_delay_t0.setValue(self.delay.time_zero)
        self.update_d_time_box_limits()
        
    def update_num_shots(self) -> None:
        if self.idle is True:
            self.num_shots = self.ui.a_num_shots.value()
            self.ui.d_num_shots.setValue(self.num_shots)

    def update_d_num_shots(self) -> None:
        if self.idle is True:
            self.num_shots = self.ui.d_num_shots.value()
            self.ui.a_num_shots.setValue(self.num_shots)

    def update_num_sweeps(self) -> None:
        if self.idle is True:
            self.num_sweeps = self.ui.a_num_sweeps.value()

    def update_use_calib(self) -> None:
        self.use_calib = self.ui.a_use_calib.isChecked()
        self.ui.d_use_calib.setChecked(self.use_calib)
        self.update_xlabel()

    def update_d_use_calib(self) -> None:
        self.use_calib = self.ui.d_use_calib.isChecked()
        self.ui.a_use_calib.setChecked(self.use_calib)
        self.update_xlabel()

    def update_d_dcshotfactor(self) -> None:
        self.dcshotfactor = self.ui.d_dcshotfactor_sb.value()

    def update_xlabel(self) -> None:
        self.xlabel = 'Wavelength (nm)' if self.use_calib else 'Pixel Number'
        self.ui.a_last_shot_graph.plotItem.setLabels(bottom=self.xlabel)
        self.ui.a_spectra_graph.plotItem.setLabels(bottom=self.xlabel)
        self.ui.d_last_shot_graph.plotItem.setLabels(bottom=self.xlabel)
        self.ui.d_error_graph.plotItem.setLabels(bottom=self.xlabel)
        self.ui.d_probe_ref_graph.plotItem.setLabels(bottom=self.xlabel)

    def update_xlabel_kinetics(self) -> None:
        label = 'Time ({0})'.format(self.timeunits)
        self.ui.a_kinetic_graph.plotItem.setLabels(bottom=label)

    def update_calib(self) -> None:
        self.calib  = [self.ui.a_calib_pixel_low.value(),
                       self.ui.a_calib_pixel_high.value(),
                       self.ui.a_calib_wave_low.value(),
                       self.ui.a_calib_wave_high.value()]
        self.ui.d_calib_pixel_low.setValue(self.calib[0])
        self.ui.d_calib_pixel_high.setValue(self.calib[1])
        self.ui.d_calib_wave_low.setValue(self.calib[2])
        self.ui.d_calib_wave_high.setValue(self.calib[3])

    def update_d_calib(self) -> None:
        self.calib  = [self.ui.d_calib_pixel_low.value(),
                       self.ui.d_calib_pixel_high.value(),
                       self.ui.d_calib_wave_low.value(),
                       self.ui.d_calib_wave_high.value()]
        self.ui.a_calib_pixel_low.setValue(self.calib[0])
        self.ui.a_calib_pixel_high.setValue(self.calib[1])
        self.ui.a_calib_wave_low.setValue(self.calib[2])
        self.ui.a_calib_wave_high.setValue(self.calib[3])

    def update_use_cutoff(self) -> None:
        self.use_cutoff = self.ui.a_use_cutoff.isChecked()
        self.ui.d_use_cutoff.setChecked(self.use_cutoff)

    def update_d_use_cutoff(self) -> None:
        self.use_cutoff = self.ui.d_use_cutoff.isChecked()
        self.ui.a_use_cutoff.setChecked(self.use_cutoff)

    def update_cutoff(self) -> None:
        if self.ui.a_cutoff_pixel_high.value() > self.ui.a_cutoff_pixel_low.value():
            self.cutoff = [self.ui.a_cutoff_pixel_low.value(),
                           self.ui.a_cutoff_pixel_high.value()]
            self.ui.d_cutoff_pixel_low.setValue(self.cutoff[0])
            self.ui.d_cutoff_pixel_high.setValue(self.cutoff[1])
        else:
            self.append_history('Cutoff Values Incompatible')

    def update_d_cutoff(self) -> None:
        if self.ui.d_cutoff_pixel_high.value() > self.ui.d_cutoff_pixel_low.value():
            self.cutoff = [self.ui.d_cutoff_pixel_low.value(),
                           self.ui.d_cutoff_pixel_high.value()]
            self.ui.a_cutoff_pixel_low.setValue(self.cutoff[0])
            self.ui.a_cutoff_pixel_high.setValue(self.cutoff[1])
        else:
            self.append_history('Cutoff Values Incompatible')

    def update_plot_log_t(self) -> None:
        self.use_logscale = self.ui.a_plot_log_t_cb.isChecked()

    def update_refman(self) -> None:
        self.refman = [self.ui.d_refman_vertical_stretch.value(),
                       self.ui.d_refman_vertical_offset.value(),
                       self.ui.d_refman_horiz_offset.value(),
                       self.ui.d_refman_scale_center.value(),
                       self.ui.d_refman_scale_factor.value()]

    def update_threshold(self) -> None:
        self.threshold = [self.ui.d_threshold_pixel.value(),
                          self.ui.d_threshold_value.value()]

    def update_d_time(self) -> None:
        self.d_time = self.ui.d_time.value()

    def update_d_jogstep(self) -> None:
        self.d_jogstep = self.ui.d_jogstep_sb.value()

    def append_history(self, message):
        self.ui.a_history.appendPlainText(message)
        self.ui.d_history.appendPlainText(message)

    def create_plots(self) -> None:
        self.ui.a_last_shot_graph.plotItem.setLabels(left='dtt', bottom=self.xlabel)
        self.ui.a_last_shot_graph.plotItem.showAxis('top', show=True)
        self.ui.a_last_shot_graph.plotItem.showAxis('right', show=True)

        self.ui.a_kinetic_graph.plotItem.setLabels(left='dtt', bottom='Time ({0})'.format(self.timeunits))
        self.ui.a_kinetic_graph.plotItem.showAxis('top', show=True)
        self.ui.a_kinetic_graph.plotItem.showAxis('right', show=True)
        
        self.ui.a_spectra_graph.plotItem.setLabels(left='dtt', bottom=self.xlabel)
        self.ui.a_spectra_graph.plotItem.showAxis('top', show=True)
        self.ui.a_spectra_graph.plotItem.showAxis('right', show=True)
        
        self.ui.d_last_shot_graph.plotItem.setLabels(left='dtt', bottom=self.xlabel) 
        self.ui.d_last_shot_graph.plotItem.showAxis('top', show=True)
        self.ui.d_last_shot_graph.plotItem.showAxis('right', show=True)
        
        self.ui.d_error_graph.plotItem.setLabels(left='Log(Error)', bottom=self.xlabel)
        self.ui.d_error_graph.plotItem.showAxis('top', show=True)
        self.ui.d_error_graph.plotItem.showAxis('right', show=True)
        
        self.ui.d_trigger_graph.plotItem.setLabels(left='Trigger Signal', bottom='Shot Number')
        self.ui.d_trigger_graph.plotItem.showAxis('top', show=True)
        self.ui.d_trigger_graph.plotItem.showAxis('right', show=True)
        
        self.ui.d_probe_ref_graph.plotItem.setLabels(left='Counts', bottom=self.xlabel)
        self.ui.d_probe_ref_graph.plotItem.showAxis('top', show=True)
        self.ui.d_probe_ref_graph.plotItem.showAxis('right', show=True)

    def set_waves_and_times_axes(self) -> None:
        self.waves = self.pixels_to_waves()
        if self.use_calib:
            self.plot_waves = self.pixels_to_waves()
        else:
            self.plot_waves = np.linspace(0, self.camera.valid_pixels-1,  self.camera.valid_pixels)
        self.plot_times = self.times
        
    def create_plot_waves_and_times(self) -> None:
        self.set_waves_and_times_axes()

        if not self.diagnostics_on:
            self.plot_kinetic_avg = self.current_sweep.avg_data[:, self.kinetics_pixel]
            self.plot_kinetic_current = self.current_sweep.current_data[:, self.kinetics_pixel]
        
        if self.diagnostics_on is False:
            self.plot_dtt = self.current_sweep.avg_data[:]
            
        self.plot_ls = self.current_data.dtt[:]
        self.plot_probe_shot_error = self.current_data.probe_shot_error[:]
        
        if self.ui.d_use_reference.isChecked() is True:
            self.plot_ref_shot_error = self.current_data.ref_shot_error[:]
            self.plot_dtt_error = self.current_data.dtt_error[:]
            
        self.plot_probe_on = self.current_data.probe_on[:]
        self.plot_reference_on = self.current_data.reference_on[:]
        self.plot_probe_on_array = self.current_data.probe_on_array[:]
        self.plot_reference_on_array = self.current_data.reference_on_array[:]
        
        if self.use_cutoff is True:
            self.plot_waves = self.plot_waves[self.cutoff[0]:self.cutoff[1]]
            
            if self.diagnostics_on is False:
                self.plot_dtt = self.plot_dtt[:,self.cutoff[0]:self.cutoff[1]]
                
            self.plot_ls = self.plot_ls[self.cutoff[0]:self.cutoff[1]]
            self.plot_probe_shot_error = self.plot_probe_shot_error[self.cutoff[0]:self.cutoff[1]]
            
            if self.ui.d_use_reference.isChecked() is True:
                self.plot_ref_shot_error = self.plot_ref_shot_error[self.cutoff[0]:self.cutoff[1]]
                self.plot_dtt_error = self.plot_dtt_error[self.cutoff[0]:self.cutoff[1]]
                
            self.plot_probe_on = self.plot_probe_on[self.cutoff[0]:self.cutoff[1]]
            self.plot_reference_on = self.plot_reference_on[self.cutoff[0]:self.cutoff[1]]
            self.plot_probe_on_array = self.plot_probe_on_array[:,self.cutoff[0]:self.cutoff[1]]
            self.plot_reference_on_array = self.plot_reference_on_array[:,self.cutoff[0]:self.cutoff[1]]        

    def pixels_to_waves(self) -> np.ndarray:
        slope = (self.calib[3]-self.calib[2])/(self.calib[1]-self.calib[0])
        y_int = self.calib[2]-slope*self.calib[0]
        return np.linspace(0, self.camera.valid_pixels-1, self.camera.valid_pixels)*slope+y_int

    def ls_plot(self) -> None:
        self.ui.a_last_shot_graph.plotItem.plot(self.plot_waves, self.plot_ls, clear=True, pen='b')

    def top_plot(self) -> None:
        self.ui.a_colourmap.setImage(self.plot_dtt, scale=(len(self.plot_waves)/len(self.times), 1))

    def add_time_marker(self) -> None:
        finite_times = self.plot_times[np.isfinite(self.plot_times)]
        self.time_marker = pg.InfiniteLine(finite_times[int(len(finite_times)/2)], movable=True, bounds=[min(self.plot_times), max(self.plot_times)])
        self.ui.a_kinetic_graph.addItem(self.time_marker)
        self.time_marker.sigPositionChangeFinished.connect(self.update_time_pixel)
        self.time_marker_label = pg.InfLineLabel(self.time_marker, text='{value:.2f}'+self.timeunits, movable=True, position=0.9)
        self.update_time_pixel()

    def update_time_pixel(self) -> None:
        self.spectrum_time = self.time_marker.value()
        self.time_pixel = np.where((self.plot_times-self.spectrum_time)**2 == min((self.plot_times-self.spectrum_time)**2))[0][0]
        if self.finished_acquisition:
            self.create_plot_waves_and_times()
            self.spec_plot()
        return
    
    def add_wavelength_marker(self) -> None:
        self.wavelength_marker = pg.InfiniteLine(self.plot_waves[int(len(self.plot_waves)/2)], movable=True, bounds=[self.plot_waves[0], self.plot_waves[-1]])
        self.ui.a_spectra_graph.addItem(self.wavelength_marker)
        self.wavelength_marker.sigPositionChangeFinished.connect(self.update_kinetics_wavelength)
        self.wavelength_marker_label = pg.InfLineLabel(self.wavelength_marker, text='{value:.2f}nm', movable=True, position=0.9)
        self.update_kinetics_wavelength()

    def update_kinetics_wavelength(self) -> None:
        self.kinetics_wavelength = self.wavelength_marker.value()
        self.kinetics_pixel = np.where((self.waves-self.kinetics_wavelength)**2 == min((self.waves-self.kinetics_wavelength)**2))[0][0]  # self.waves rather than self.plot_waves?
        if self.finished_acquisition:
            self.create_plot_waves_and_times()
            self.kin_plot()

    def kin_plot(self) -> None:
        for item in self.ui.a_kinetic_graph.plotItem.listDataItems():
            self.ui.a_kinetic_graph.plotItem.removeItem(item)
        if self.finished_acquisition:
            self.ui.a_kinetic_graph.plotItem.plot(self.plot_times, self.plot_kinetic_avg, pen='b', symbol='s', symbolPen='b', symbolBrush=None, symbolSize=4, clear=False)
        else:
            if self.current_sweep.sweep_index > 0:
                self.ui.a_kinetic_graph.plotItem.plot(self.plot_times[0:self.timestep+1], self.plot_kinetic_current[0:self.timestep+1], pen='c', symbol='s', symbolPen='c', symbolBrush=None, symbolSize=4, clear=False)
                self.ui.a_kinetic_graph.plotItem.plot(self.plot_times, self.plot_kinetic_avg, pen='b', symbol='s', symbolPen='b', symbolBrush=None, symbolSize=4, clear=False)
            else:
                self.ui.a_kinetic_graph.plotItem.plot(self.plot_times[0:self.timestep+1], self.plot_kinetic_current[0:self.timestep+1], pen='c', symbol='s', symbolPen='c', symbolBrush=None, symbolSize=4, clear=False)

    def spec_plot(self) -> None:
        for item in self.ui.a_spectra_graph.plotItem.listDataItems():
            self.ui.a_spectra_graph.plotItem.removeItem(item)
        self.ui.a_spectra_graph.plotItem.plot(self.plot_waves, self.plot_dtt[self.time_pixel,:], pen='r', clear=False)

    def d_error_plot(self) -> None:
        self.ui.d_error_graph.plotItem.plot(self.plot_waves, np.log10(self.plot_probe_shot_error), pen='r', clear=True, fillBrush='r')
        if self.ui.d_use_reference.isChecked() is True:         
            self.ui.d_error_graph.plotItem.plot(self.plot_waves, np.log10(self.plot_ref_shot_error), pen='g', clear=False, fillBrush='g')    
            self.ui.d_error_graph.plotItem.plot(self.plot_waves, np.log10(self.plot_dtt_error), pen='b', clear=False, fillBrush='b')
        self.ui.d_error_graph.plotItem.setYRange(-4, 1, padding=0)

    def d_trigger_plot(self) -> None:
        self.ui.d_trigger_graph.plotItem.plot(np.arange(self.num_shots), self.current_data.trigger, pen=None, symbol='o', clear=True)

    def d_probe_ref_plot(self) -> None:
        for item in self.ui.d_probe_ref_graph.plotItem.listDataItems():
            self.ui.d_probe_ref_graph.plotItem.removeItem(item)
        probe_std = np.std(self.plot_probe_on_array, axis=0)
        self.ui.d_probe_ref_graph.plotItem.plot(self.plot_waves, self.plot_probe_on, pen='r')
        pcurve1 = pg.PlotDataItem(self.plot_waves, self.plot_probe_on-2*probe_std, pen='r')
        pcurve2 = pg.PlotDataItem(self.plot_waves, self.plot_probe_on+2*probe_std, pen='r')
        self.probe_error_region.setCurves(pcurve1, pcurve2)
        self.ui.d_probe_ref_graph.addItem(self.probe_error_region)            
        if self.ui.d_use_reference.isChecked() is True:
            ref_std = np.std(self.plot_reference_on_array, axis=0)
            self.ui.d_probe_ref_graph.plotItem.plot(self.plot_waves, self.plot_reference_on, pen='b')
            rcurve1 = pg.PlotDataItem(self.plot_waves, self.plot_reference_on-2*ref_std, pen='b')
            rcurve2 = pg.PlotDataItem(self.plot_waves, self.plot_reference_on+2*ref_std, pen='b')
            self.ref_error_region.setCurves(rcurve1, rcurve2)
            self.ui.d_probe_ref_graph.addItem(self.ref_error_region)

    def d_ls_plot(self) -> None:
        self.ui.d_last_shot_graph.plotItem.plot(self.plot_waves, self.plot_ls, pen='b', clear=True)

    @staticmethod
    def message_box(text: str, info: str | None = None) -> int:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setText(text)
        if info:
            msg.setInformativeText(info)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        retval = msg.exec()
        return retval
        
    def running(self) -> None:
        self.idle = False
        self.ui.hardware_tab.setEnabled(False)
        self.ui.a_run_btn.setDisabled(True)
        self.ui.d_run_btn.setDisabled(True)
        self.ui.a_file_box.setDisabled(True)
        self.ui.a_times_box.setDisabled(True)
        self.ui.a_acquire_box.setDisabled(True)
        self.ui.a_calib_box.setDisabled(True)
        self.ui.a_cutoff_box.setDisabled(True)
        if self.diagnostics_on is False:
            self.ui.d_times_box.setDisabled(True)
            self.ui.d_other_box.setDisabled(True)
            self.ui.d_calib_box.setDisabled(True)
            self.ui.d_cutoff_box.setDisabled(True)
            self.ui.d_refmanip_box.setDisabled(True)
            self.ui.d_acquire_box.setDisabled(True)
            
    def idling(self) -> None:
        self.idle = True
        self.ui.hardware_tab.setEnabled(True)
        self.ui.a_run_btn.setDisabled(False)
        self.ui.d_run_btn.setDisabled(False)
        self.ui.a_file_box.setDisabled(False)
        self.ui.a_times_box.setDisabled(False)
        self.ui.a_acquire_box.setDisabled(False)
        self.ui.a_calib_box.setDisabled(False)
        self.ui.a_cutoff_box.setDisabled(False)
        self.ui.d_refmanip_box.setDisabled(False)
        self.ui.d_acquire_box.setDisabled(False)
        self.ui.d_other_box.setDisabled(False)
        self.ui.d_calib_box.setDisabled(False)
        self.ui.d_cutoff_box.setDisabled(False)
        self.ui.d_times_box.setDisabled(False)
    
    def update_progress_bars(self) -> None:
        self.ui.a_sweep_progress_bar.setValue(self.timestep+1)
        self.ui.a_measurement_progress_bar.setValue((len(self.times)*self.current_sweep.sweep_index)+self.timestep+1)
        
    def acquire(self) -> None:
        self.append_history(f'Acquiring{self.num_shots} shots')
        self.acquisition_requested.emit()
    
    @pyqtSlot(np.ndarray, np.ndarray, int, int)    
    def post_acquire(self, probe: np.ndarray, reference: np.ndarray, first_pixel: int, num_pixels: int) -> None:
        self.acquisition_processing_requested.emit(probe, reference, first_pixel, num_pixels, self.threshold[0], self.threshold[1], self.tau_flip_request, True, True)

    @pyqtSlot()
    def post_acquisition_processing(self) -> None:
        self.current_data = self.acquisition.data
        if not self.acquisition.data.high_trigger_std and not self.acquisition.data.high_dtt:
            self.current_sweep.add_current_data(dtt=self.current_data.dtt, time_point=self.timestep)
            self.create_plot_waves_and_times()
            if self.ui.acquisition_tab.isVisible() is True:
                self.ls_plot()
                self.top_plot()
                self.kin_plot()
                self.spec_plot()
            if self.ui.diagnostics_tab.isVisible() is True:
                self.d_ls_plot()
                self.d_error_plot()
                self.d_trigger_plot()
                self.d_probe_ref_plot()
            if self.stop_request is True:
                self.finish()
            if self.timestep == len(self.times)-1:
                self.post_sweep()
            else:
                self.timestep = self.timestep+1
                time = self.times[self.timestep]
                self.ui.a_time_display.display(time)
                self.update_progress_bars()
                self.move_delay_requested.emit(time)
                self.acquire()
        else:
            if self.stop_request is True:
                self.finish()
            self.append_history('retaking point')
            self.acquire()
   
    def acquire_bgd(self) -> None:
        self.append_history(f'Acquiring {self.num_shots*self.dcshotfactor} shots')
        self.acquisition_requested.emit()
    
    @pyqtSlot(np.ndarray, np.ndarray, int, int)    
    def post_acquire_bgd(self, probe: np.ndarray, reference: np.ndarray, first_pixel: int, num_pixels: int) -> None:
        self.message_box(text="Unblock probe and reference beams")
        self.acquisition_background_processing_requested.emit(probe, reference, first_pixel, num_pixels, self.threshold[0], self.threshold[1])

    @pyqtSlot()
    def post_background_processing(self) -> None:
        self.run()          

    def exec_run_btn(self) -> None:
        if self.ui.a_test_run_btn.isChecked() is True:
            self.append_history('Launching Test Run!')
        else:
            self.append_history('Launching Run!')
        
        self.stop_request = False
        self.diagnostics_on = False
        self.running()
        self.update_num_shots()
        
        if not self.delay.check_times(self.times):
            self.message_box(text="One or more time point exceeds physical limit of delay!")
            self.idling()
            return
        
        self.finished_acquisition = False
        self.set_waves_and_times_axes()
        try:
            self.ui.a_kinetic_graph.removeItem(self.time_marker)
        except:
            pass
        self.add_time_marker()
        try:
            self.ui.a_spectra_graph.removeItem(self.wavelength_marker)
        except:
            pass
        self.add_wavelength_marker()

        self.camera.acquisition_finished.connect(self.post_acquire_bgd)

        self.acquisition.bgd_processing_finished.connect(self.post_background_processing)
        self.acquisition.processing_finished.connect(self.post_acquisition_processing)

        self.processing_thread.start()
        
        if not self.ui.a_test_run_btn.isChecked():
            self.message_box(text="Block probe and reference beams")
            self.append_history('Taking Background')
            self.acquire_bgd()
        else:
            self.run()

    def run(self) -> None:
        self.update_metadata()
        self.current_sweep = Sweep(self.times, self.camera.valid_pixels, self.filepath, self.metadata)
        
        self.camera.update_number_of_scans(self.num_shots)

        self.camera.acquisition_finished.disconnect(self.post_acquire_bgd)
        self.camera.acquisition_finished.connect(self.post_acquire)

        self.delay.move_finished.connect(self.post_delay_moved)
        self.save_sweep_requested.connect(self.current_sweep.save)
        self.current_sweep.data_saved.connect(self.post_sweep_saved)
        self.current_sweep.data_saved.connect(self.save_thread.quit)

        self.append_history(f'Starting sweep {self.current_sweep.sweep_index}')
        self.ui.a_sweep_display.display(self.current_sweep.sweep_index+1)
        self.ui.a_sweep_progress_bar.setMaximum(len(self.times))
        self.ui.a_measurement_progress_bar.setMaximum(len(self.times)*self.num_sweeps)
        self.start_sweep()
        
    def finish(self) -> None:
        self.idling()
        self.finished_acquisition = True
        if not self.stop_request:
            self.create_plot_waves_and_times()
            if self.ui.acquisition_tab.isVisible():
                self.ls_plot()
                self.top_plot()
                self.kin_plot()
                self.spec_plot()
            if self.ui.diagnostics_tab.isVisible():
                self.d_ls_plot()
                self.d_error_plot()
                self.d_trigger_plot()
                self.d_probe_ref_plot()
        self.processing_thread.quit()

    def start_sweep(self) -> None:
        self.timestep = 0
        time = self.times[self.timestep]
        self.ui.a_time_display.display(time)
        self.update_progress_bars()
        self.move_delay_requested.emit(time)

    @pyqtSlot(bool)
    def post_delay_moved(self, tflip: bool) -> None:
        self.tau_flip_request = tflip
        self.acquire()

    def post_sweep(self) -> None:
        if not self.ui.a_test_run_btn.isChecked():
            self.append_history(f'Saving sweep {self.current_sweep.sweep_index}')

            self.current_sweep.moveToThread(self.save_thread)
            self.save_thread.start()
            self.save_sweep_requested.emit(self.waves, self.current_data.probe_on, self.current_data.reference_on, self.current_data.probe_shot_error)
        else:
            self.post_sweep_saved()

    @pyqtSlot()
    def post_sweep_saved(self) -> None:
        self.current_sweep.next_sweep()
        if self.current_sweep.sweep_index == self.num_sweeps:
            self.finish()
        else:
            self.append_history(f'Starting sweep {self.current_sweep.sweep_index}')
            self.ui.a_sweep_display.display(self.current_sweep.sweep_index+1)
            self.start_sweep()
        
    def exec_stop_btn(self) -> None:
        self.append_history('Stopped')
        self.stop_request = True
    
    def d_jog_earlier(self) -> None:
        new_time = self.d_time-self.d_jogstep
        self.ui.d_time.setValue(new_time)
        self.move_delay_requested.emit(new_time)
        
    def d_jog_later(self) -> None:
        new_time = self.d_time+self.d_jogstep
        self.ui.d_time.setValue(new_time)
        self.move_delay_requested.emit(new_time)

    @pyqtSlot(bool)
    def d_post_delay_moved(self, tflip: bool) -> None:
        self.tau_flip_request = tflip

    def exec_d_set_current_btn(self) -> None:
        self.ui.d_delay_t0.setValue(self.delay.time_zero-self.d_time)
        self.ui.d_time.setValue(0)
        self.update_d_time()
        self.move_delay_requested.emit(self.d_time)
   
    def d_acquire(self) -> None:
        self.append_history(f'Acquiring {self.num_shots} shots')
        self.acquisition_requested.emit()
        
    @pyqtSlot(np.ndarray, np.ndarray, int, int)
    def d_post_acquire(self, probe: np.ndarray, reference: np.ndarray, first_pixel: int, num_pixels: int) -> None:
        self.acquisition_processing_requested.emit(probe, reference, first_pixel, num_pixels, self.threshold[0], self.threshold[1], self.tau_flip_request, True, True)

    @pyqtSlot()
    def d_post_acquisition_processing(self) -> None:
        self.current_data = self.acquisition.data

        self.create_plot_waves_and_times()
        self.d_ls_plot()
        self.d_error_plot()
        self.d_trigger_plot()
        self.d_probe_ref_plot()
        
        if self.stop_request is True:
            self.d_finish()
        else:
            self.d_acquire()

    def d_acquire_bgd(self) -> None:
        self.append_history(f'Acquiring {self.num_shots*self.dcshotfactor} shots')
        self.acquisition_requested.emit()
        
    @pyqtSlot(np.ndarray, np.ndarray, int, int)
    def d_post_acquire_bgd(self, probe: np.ndarray, reference: np.ndarray, first_pixel: int, num_pixels: int) -> None:
        self.message_box(text="Unblock probe and reference beams")
        self.acquisition_background_processing_requested.emit(probe, reference, first_pixel, num_pixels, self.threshold[0], self.threshold[1])

    @pyqtSlot()
    def d_post_background_processing(self) -> None:
        self.delay.move_finished.connect(self.d_run)
        self.move_delay_requested.emit(self.d_time)

    def exec_d_run_btn(self) -> None:
        self.append_history('Launching Diagnostics!')
        self.stop_request = False
        self.diagnostics_on = True
        self.tau_flip_request = False
        self.running()
        self.ui.a_test_run_btn.setChecked(0)
        self.update_d_num_shots()
        
        success = self.delay.check_time(self.d_time)
        if success is False:
            self.message_box(text="One or more time point exceeds physical limit of delay!")
            self.idling()
            return

        self.camera.acquisition_finished.connect(self.d_post_acquire_bgd)
        self.acquisition.bgd_processing_finished.connect(self.d_post_background_processing)
        self.acquisition.processing_finished.connect(self.d_post_acquisition_processing)

        self.processing_thread.start()

        self.message_box(text="Block probe and reference beams")
        self.append_history('Taking Background')
        self.d_acquire_bgd()

    @pyqtSlot(bool)
    def d_run(self, tflip: bool) -> None:
        self.tau_flip_request = tflip
        self.camera.update_number_of_scans(self.num_shots)
        self.camera.acquisition_finished.disconnect(self.d_post_acquire_bgd)
        self.camera.acquisition_finished.connect(self.d_post_acquire)

        self.delay.move_finished.disconnect(self.d_run)
        self.delay.move_finished.connect(self.d_post_delay_moved)

        self.d_acquire()
        
    def d_finish(self) -> None:
        self.idling()
        self.processing_thread.quit()
        
    def exec_d_stop_btn(self) -> None:
        self.stop_request = True

    def exec_d_move_to_time(self) -> None:
        self.move_delay_requested.emit()


def main() -> None:
    
    # create application
    QtWidgets.QApplication.setStyle('Fusion')
    app = QtWidgets.QApplication(sys.argv)
    
    # load the parameter values from last time and launch GUI
    last_instance_filename = 'last_instance_values.txt'
    # last_instance_values = pd.read_csv(last_instance_filename, sep=':', header=None, index_col=0).squeeze("columns")

    last_instance_values = None

    ex = Application(last_instance_filename, last_instance_values=last_instance_values, preloaded=False)
    
    ex.show()
    ex.create_plots()
    
    # kill application
    sys.exit(app.exec())
   

if __name__ == '__main__':
    main()
