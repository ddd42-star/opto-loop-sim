from registry import device_registry
from pymmcore_stage_sim import SimStageDevice

@device_registry("XYStageDevice")
def make_xy_stage(microscope_sim):
    return SimStageDevice(microscope_sim)

