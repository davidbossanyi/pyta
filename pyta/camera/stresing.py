from __future__ import annotations

import ctypes as ct
from typing import Any

import numpy as np

from pyta.camera.base import ICamera


class VisCamera(ICamera):
    def __init__(self, dll_path: str):
        super().__init__()
        self.dll = ct.WinDLL(dll_path)  # type: ignore[attr-defined]
        self.board_number = 1  # PCI board index: 1 is VIS, 2 is NIR
        self.fft_lines = 64  # number of lines for binning if FFT sensor, 0 for NIR, 64 for VIS
        self.vfreq = 7  # vertical frequency for FFT sensor, given as 7 in examples from Stresing
        self._total_pixels = 1200  # number of pixels, including dummy pixels
        self._valid_pixels = 1024  # actual number of active pixels
        self._first_pixel = 16  # first non-dummy pixel
        self.threadp = 15  # priority of thread, 31 is highest
        self.dat_10ns = 100  # delay after trigger
        self.zadr = 1  # not needed, only if in addressed mode
        self.fkt = 1  # 1 for standard read, others are possible, could try 0 but unlikely
        self.sym = 0  # for FIFO, depends on sensor
        self.burst = 1  # for FIFO, depends on sensor
        self.waits = 3  # depends on sensor, sets the pixel read frequency
        self.flag816 = 1  # 1 if AD resolution 12 is 16bit, 2 if 8bit
        self.pportadr = 378  # address if parallel port is used
        self.pclk = 2  # pixelclock, not used here
        self.xckdelay = 3  # sets the delay after xck goes high
        self.freq = 0  # read frequency in Hz, should be 0 if exposure time is given
        self.clear_cnt = 8  # number of reads to clear the sensor, depends on sensor
        self.release_ms = -1  # less than zero: don't release
        self.exttrig = 1  # 1 is use external trigger
        self.block_trigger = 0  # true (not 0) if one external trigger starts block of nos scans which run with internal timer
        self.adrdelay = 3  # not sure what this is...
        self._exposure_time_us = 1
        self._number_of_scans = 100
        self.array = np.zeros((self.number_of_scans + 10, self.total_pixels * 2), dtype=np.dtype(np.int32))
        self._set_argtypes()

    def __str__(self) -> str:
        return "Stresing VIS"

    def update_number_of_scans(self, number_of_scans: int) -> None:
        self.number_of_scans = number_of_scans
        self.array = np.zeros((self.number_of_scans + 10, self.total_pixels * 2), dtype=np.dtype(np.int32))

    @property
    def first_pixel(self) -> int:
        return self._first_pixel

    @property
    def data(self) -> np.ndarray:
        return self.array[10:]

    @property
    def total_pixels(self) -> int:
        return self._total_pixels

    @property
    def valid_pixels(self) -> int:
        return self._valid_pixels

    def _set_argtypes(self) -> None:
        self.dll.DLLReadFFLoop.argtypes = [
            ct.c_uint32,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags=["C", "W"]),
            ct.c_uint32,
            ct.c_int32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint32,
            ct.c_uint16,
            ct.c_uint8,
            ct.c_uint8,
        ]
        self.dll.DLLGETCCD.argtypes = [
            ct.c_uint32,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags=["C", "W"]),
            ct.c_uint32,
            ct.c_int32,
            ct.c_uint32,
        ]
        self.dll.DLLReadFifo.argtypes = [
            ct.c_uint32,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags=["C", "W"]),
            ct.c_int32,
        ]

    def _connect(self) -> None:
        self.CCDDrvInit()
        self._wait(1000000)
        self.InitBoard()
        self._wait(1000000)
        self.WriteLongS0(100, 52)
        self._wait(1000000)
        self.RsTOREG()
        self._wait(1000000)
        self.SetISFFT(1)
        self._wait(1000000)
        self.SetupVCLK()
        self._wait(1000000)
        self.Cal16bit()
        self._wait(1000000)
        self.RSFifo()
        self._wait(1000000)

    def _wait(self, time_us: int) -> None:
        self.InitSysTimer()
        tick_start = self.TicksTimestamp()
        time_start = self.Tickstous(tick_start)
        tick_end = self.TicksTimestamp()
        time_end = self.Tickstous(tick_end)
        while (time_end - time_start) < time_us:
            tick_end = self.TicksTimestamp()
            time_end = self.Tickstous(tick_end)

    def _read(self) -> None:
        self.ReadFFLoop(self.number_of_scans, self.exposure_time_us)
        self._construct_data_vectors()

    def _construct_data_vectors(self) -> None:
        hilo_array = self.data.view(np.uint16)[:, 0 : self.total_pixels * 2]
        hilo_array = hilo_array.reshape(hilo_array.shape[0], 2, self.total_pixels)
        self._probe = hilo_array[:, 0, :]  # type: ignore[assignment]
        self._reference = hilo_array[:, 1, :]  # type: ignore[assignment]

    def _disconnect(self) -> None:
        self.CCDDrvExit()

    def _overflow(self) -> None:
        self.FFOvl()

    ###########################################################################
    ###########################################################################
    ###########################################################################
    # Library methods from DLL (DO NOT EDIT)

    def AboutDrv(self) -> None:
        self.dll.DLLAboutDrv(ct.c_uint32(self.board_number))

    def ActCooling(self) -> None:
        self.dll.DLLActCooling(ct.c_uint32(self.board_number), ct.c_uint8(1))

    def ActMouse(self) -> None:
        self.dll.DLLActMouse(ct.c_uint32(self.board_number))

    def Cal16bit(self) -> None:
        self.dll.DLLCal16Bit(ct.c_uint32(self.board_number), ct.c_uint32(self.zadr))

    def CCDDrvExit(self) -> None:
        self.dll.DLLCCDDrvExit(ct.c_uint32(self.board_number))

    def CCDDrvInit(self) -> bool:
        found = self.dll.DLLCCDDrvInit(ct.c_uint32(self.board_number))
        return bool(found)

    def CloseShutter(self) -> None:
        self.dll.DLLCloseShutter(ct.c_uint32(self.board_number))

    def ClrRead(self, clr_count: int) -> None:
        self.dll.DLLClrRead(
            ct.c_uint32(self.board_number), ct.c_uint32(self.fft_lines), ct.c_uint32(self.zadr), ct.c_uint32(clr_count)
        )

    def ClrShCam(self) -> None:
        self.dll.DLLClrShCam(ct.c_uint32(self.board_number), ct.c_uint32(self.zadr))

    def DeactMouse(self) -> None:
        self.dll.DLLDeactMouse(ct.c_uint32(self.board_number))

    def DisableFifo(self) -> None:
        self.dll.DLLDisableFifo(ct.c_uint32(self.board_number))

    def EnableFifo(self) -> None:
        self.dll.DLLEnableFifo(ct.c_uint32(self.board_number))

    def FFOvl(self) -> bool:
        overflow = self.dll.DLLFFOvl(ct.c_uint32(self.board_number))
        return bool(overflow)

    def FFValid(self) -> bool:
        valid = self.dll.DLLFFValid(ct.c_uint32(self.board_number))
        return bool(valid)

    def FlagXCKI(self) -> bool:
        active = self.dll.DLLFlagXCKI(ct.c_uint32(self.board_number))
        return bool(active)

    def GetCCD(self) -> np.ndarray:
        self.dll.DLLGETCCD(
            ct.c_uint32(self.board_number),
            self.array,
            ct.c_uint32(self.fft_lines),
            ct.c_int32(self.fkt),
            ct.c_uint32(self.zadr),
        )
        return self.array

    def HighSlope(self) -> None:
        self.dll.DLLHighSlope(ct.c_uint32(self.board_number))

    def InitBoard(self) -> None:
        self.dll.DLLInitBoard(
            ct.c_uint32(self.board_number),
            ct.c_int8(self.sym),
            ct.c_uint8(self.burst),
            ct.c_uint32(self.total_pixels),
            ct.c_uint32(self.waits),
            ct.c_uint32(self.flag816),
            ct.c_uint32(self.pportadr),
            ct.c_uint32(self.pclk),
            ct.c_uint32(self.adrdelay),
        )

    def InitSysTimer(self) -> Any:
        return self.dll.DLLInitSysTimer()

    def LowSlope(self) -> None:
        self.dll.DLLLowSlope(ct.c_uint32(self.board_number))

    def OpenShutter(self) -> None:
        self.dll.DLLOpenShutter(ct.c_uint32(self.board_number))

    def OutTrigHigh(self) -> None:
        self.dll.DLLOutTrigHigh(ct.c_uint32(self.board_number))

    def OutTrigLow(self) -> None:
        self.dll.DLLOutTrigLow(ct.c_uint32(self.board_number))

    def OutTrigPulse(self, pulse_width: int) -> None:
        self.dll.DLLOutTrigPulse(ct.c_uint32(self.board_number), ct.c_uint32(pulse_width))

    def ReadFifo(self) -> np.ndarray:
        self.dll.DLLReadFifo(ct.c_uint32(self.board_number), self.array, ct.c_int32(self.fkt))
        return self.array

    def ReadFFCounter(self) -> int:
        counter = self.dll.DLLReadFFCounter(ct.c_uint32(self.board_number))
        return counter

    def ReadFFLoop(self, number_of_scans: int, exposure_time_us: int) -> None:
        self.dll.DLLReadFFLoop(
            ct.c_uint32(self.board_number),
            self.array,
            ct.c_uint32(self.fft_lines),
            ct.c_int32(self.fkt),
            ct.c_uint32(self.zadr),
            ct.c_uint32(number_of_scans + 10),
            ct.c_uint32(exposure_time_us),
            ct.c_uint32(self.freq),
            ct.c_uint32(self.threadp),
            ct.c_uint32(self.clear_cnt),
            ct.c_uint16(self.release_ms),
            ct.c_uint8(self.exttrig),
            ct.c_uint8(self.block_trigger),
        )

    def RSFifo(self) -> None:
        self.dll.DLLRSFifo(ct.c_uint32(self.board_number))

    def RsTOREG(self) -> None:
        self.dll.DLLRsTOREG(ct.c_uint32(self.board_number))

    def SetADAmpRed(self, gain: int) -> None:
        self.dll.DLLSetADAmpRed(ct.c_uint32(self.board_number), ct.c_uint32(gain))

    def SetAD16Default(self) -> None:
        self.dll.DLLSetAD16Default(ct.c_uint32(self.board_number), ct.c_uint32(1))

    def SetExtTrig(self) -> None:
        self.dll.DLLSetExtTrig(ct.c_uint32(self.board_number))

    def StopFFTimer(self) -> None:
        self.dll.DLLStopFFTimer(ct.c_uint32(self.board_number))

    def SetIntTrig(self) -> None:
        self.dll.DLLSetIntTrig(ct.c_uint32(self.board_number))

    def SetISFFT(self, _set: int) -> None:
        self.dll.DLLSetISFFT(ct.c_uint32(self.board_number), ct.c_uint8(_set))

    def SetISPDA(self, _set: int) -> None:
        self.dll.DLLSetISPDA(ct.c_uint32(self.board_number), ct.c_uint8(_set))

    def SetOvsmpl(self) -> None:
        self.dll.DLLSetOvsmpl(ct.c_uint32(self.board_number), ct.c_uint32(self.zadr))

    def SetTemp(self, level: int) -> None:
        self.dll.DLLSetTemp(ct.c_uint32(self.board_number), ct.c_uint32(level))

    def SetupDelay(self, delay: int) -> None:
        self.dll.DLLSetupDELAY(ct.c_uint32(self.board_number), ct.c_uint32(delay))

    def SetupHAModule(self, fft_lines: int) -> None:
        self.dll.DLLSetupHAModule(ct.c_uint32(self.board_number), ct.c_uint32(fft_lines))

    def SetupVCLK(self) -> None:
        self.dll.DLLSetupVCLK(ct.c_uint32(self.board_number), ct.c_uint32(self.fft_lines), ct.c_uint8(self.vfreq))

    def StartTimer(self, exposure_time: int) -> None:
        self.dll.DLLStartTimer(ct.c_uint32(self.board_number), ct.c_uint32(exposure_time))

    def TempGood(self, channel: int) -> None:
        self.dll.DLLTempGood(ct.c_uint32(self.board_number), ct.c_uint32(channel))

    def TicksTimestamp(self) -> int:
        ticks = self.dll.DLLTicksTimestamp()
        return ticks

    def Tickstous(self, ticks: int) -> int:
        us = self.dll.DLLTickstous(ct.c_uint64(ticks))
        return us

    def Von(self) -> None:
        self.dll.DLLVon(ct.c_uint32(self.board_number))

    def Voff(self) -> None:
        self.dll.DLLVoff(ct.c_uint32(self.board_number))

    def WaitforTelapsed(self, t_us: int) -> bool:
        success = self.dll.DLLWaitforTelapsed(ct.c_uint32(t_us))
        return bool(success)

    def WriteLongS0(self, val: int, offset: int) -> None:
        self.dll.DLLWriteLongS0(ct.c_uint32(self.board_number), ct.c_uint32(val), ct.c_uint32(offset))
