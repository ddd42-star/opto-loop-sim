import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable
from pymmcore_slm_sim import SimSLMDevice
import numpy as np
from numpy.typing import DTypeLike
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from microscope_sim import MicroscopeSim
import pygame

class SimCameraDevice(CameraDevice):
    """
    pymmcore_camera_sim.py

    A virtual camera device for pymmcore that generates images using the microscope_sim.py simulation.
    """
    _exposure: float = 10.0
    _mask: np.ndarray | None = None

    def __init__(self, core: UniMMCore | None = None, microscope_sim: MicroscopeSim | None = None) -> None:
        super().__init__()
        if microscope_sim is not None:
            self._sim = microscope_sim
        else:
            # Create a new instance of the microscope simulation with default parameters
            self._sim = MicroscopeSim()
        if core is None:
            print("Note: Provide core to the SimCameraDevice constructor to use SLM features")
        self._core = core
        self._mask = None

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure

    def shape(self) -> tuple[int, int]:
        # Use the simulation's dimensions
        return (self._sim.height, self._sim.width)

    def dtype(self) -> DTypeLike:
        return np.uint8

    def set_mask(self, mask: np.ndarray | None) -> None:
        self._mask = mask

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:

        count = 0
        while n is None or count < n:
            time.sleep(self._exposure / 1000.0)
            buf = get_buffer(self.shape(), self.dtype())
            # Try to read the mask from the core SLM device, if available.
            mask = None
            if self._core is not None:
                try:
                    slm_device = self._core.getSLMDevice()
                    if slm_device:
                        mask = self._core.getSLMImage(slm_device)
                        if mask is not None:
                            mask = mask.astype(bool)
                        else:
                            print("SLM device returned no image, using default mask.")
                    else:
                        print("No SLM device found in core.")
                except Exception as e:
                    print(f"Error getting SLM image: {e}")
            if mask is None:
                mask = np.zeros((self._sim.height, self._sim.width), dtype=bool)
            self._mask = mask
            surf = self._sim.get_frame(self._mask) 
            arr = pygame.surfarray.array3d(surf)
            # Convert to grayscale (take one channel)
            arr = arr[..., 0].astype(np.uint8)
            buf[:] = arr.T  # Transpose to (height, width)
            yield {
                "data": buf,
                "timestamp": time.time()
                }
            # update count
            count += 1
# self._seq_buffer.finalize_slot(
#                 {
#                     **base_meta,
#                     **cam_meta,
#                     KW.Metadata_TimeInCore: received,
#                     KW.Metadata_ImageNumber: str(img_number),
#                     KW.Elapsed_Time_ms: f"{elapsed_ms:.2f}",
#                 }
#             )


def test():
    # Example usage
    core = UniMMCore()
    core.loadPyDevice("Camera", SimCameraDevice())
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")
    core.setExposure(42)

    try:
        from pymmcore_widgets import ExposureWidget, ImagePreview, LiveButton, SnapButton
        from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

        app = QApplication([])

        window = QWidget()
        window.setWindowTitle("Sim Microscope Camera Example")
        layout = QVBoxLayout(window)

        top = QHBoxLayout()
        top.addWidget(SnapButton(mmcore=core))
        top.addWidget(LiveButton(mmcore=core))
        top.addWidget(ExposureWidget(mmcore=core))
        layout.addLayout(top)
        layout.addWidget(ImagePreview(mmcore=core))
        window.setLayout(layout)
        window.resize(800, 600)
        window.show()
        app.exec()
    except Exception:
        print("run `pip install pymmcore-widgets[image] PyQt6` to run the GUI example")
        core.snapImage()
        image = core.getImage()
        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)
        print("Image data:", image)