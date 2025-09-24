from registry import device_registry
from pymmcore_state_device_sim import SimStateDevice

@device_registry("LEDState")
def make_led(label, state_label, microscope_sim):
    return SimStateDevice(label, state_label, microscope_sim)

@device_registry("FWState")
def make_filter_wheel(label, state_label, microscope_sim):
    return SimStateDevice(label, state_label, microscope_sim)