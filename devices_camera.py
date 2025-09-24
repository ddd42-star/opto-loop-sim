from registry import device_registry
from pymmcore_camera_sim import SimCameraDevice

@device_registry("CameraDevice")
def make_camera(core, microscope_simulation):
    return SimCameraDevice(core, microscope_simulation)
