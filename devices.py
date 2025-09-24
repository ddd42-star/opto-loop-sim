from registry import device_registry
from pymmcore_camera_sim import SimCameraDevice
from pymmcore_shutter_sim import SimShutterDevice
from pymmcore_stage_sim import SimStageDevice
from pymmcore_state_device_sim import SimStateDevice

@device_registry("CameraDevice")
def make_camera(core, microscope_simulation):
    return SimCameraDevice(core, microscope_simulation)


@device_registry("ShutterDevice")
def make_shutter():
    return SimShutterDevice()


@device_registry("LEDState")
def make_led(label, state_label, microscope_sim):
    return SimStateDevice(label, state_label, microscope_sim)

@device_registry("FWState")
def make_filter_wheel(label, state_label, microscope_sim):
    return SimStateDevice(label, state_label, microscope_sim)


@device_registry("XYStageDevice")
def make_xy_stage(microscope_sim):
    return SimStageDevice(microscope_sim)