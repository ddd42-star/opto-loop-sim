from registry import device_registry
from pymmcore_shutter_sim import SimShutterDevice


@device_registry("ShutterDevice")
def make_shutter():
    return SimShutterDevice()