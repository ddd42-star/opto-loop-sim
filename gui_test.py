from pymmcore_widgets import ExposureWidget, ImagePreview, LiveButton, SnapButton,StageWidget
from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
from pymmcore_camera_sim import SimCameraDevice
from pymmcore_slm_sim import SimSLMDevice
from pymmcore_stage_sim import SimStageDevice
from pymmcore_state_device_sim import SimStateDevice
from pymmcore_shutter_sim import SimShutterDevice
from pymmcore_z_stage_sim import SimZStageDevice
from pymmcore_plus.experimental.unicore import UniMMCore
from microscope_sim import MicroscopeSim
from pymmcore_plus import CMMCorePlus, Keyword
from pathlib import Path


core = UniMMCore()

microscope_simulation = MicroscopeSim()
print(core.getLoadedDevices())
#-----------------------------------------
# unload all devices
core.unloadAllDevices()

print(core.getLoadedDevices())
#-----------------------------------------------------
# load device
core.loadPyDevice("Camera", SimCameraDevice(core=core, microscope_sim=microscope_simulation))
#core.loadPyDevice("XYStage", SimStageDevice(microscope_sim=microscope_simulation))
core.loadPyDevice("ZStage", SimZStageDevice(microscope_sim=microscope_simulation))
core.loadPyDevice("LED", SimStateDevice(label="LED",state_dict={0:"UV", 1:"BLUE", 2:"CYAN", 3:"GREEN", 4:"YELLOW", 5:"ORANGE", 6:"RED"}, microscope_sim=microscope_simulation))
core.loadPyDevice("Filter Wheel", SimStateDevice(label="Filter Wheel",state_dict={0:"Electra1(402/454)", 1:"SCFP2(434/474)", 2:"TagGFP2(483/506)", 3:"obeYFP(514/528)", 5:"mRFP1-Q667(549/570)", 6:"mScarlet3(569/582)", 7:"miRFP670(642/670)"}, microscope_sim=microscope_simulation))
core.loadPyDevice("Shutter", SimShutterDevice())
#------------------------------------
print(core.getLoadedDevices())
# initialize device
core.initializeDevice("Camera")
#core.initializeDevice("XYStage")
core.initializeDevice("Filter Wheel")
core.initializeDevice("LED")
core.initializeDevice("Shutter")
core.initializeDevice("ZStage")

#core.initializeAllDevices()

print(core.getLoadedDevices())

print(core.getDeviceInitializationState("LED"))

#---------------------------
# Set initial value of some device
core.setCameraDevice("Camera")
#core.setXYStageDevice("XYStage")
core.setFocusDevice("ZStage")
core.setShutterDevice("Shutter")
core.setState("LED", 0)
core.setState("Filter Wheel", 0)

core.defineConfigGroup("Fake")
core.defineConfigGroup("Real")
core.defineConfig("Fake", "nucleus-channel", "LED", "Label", "ORANGE")
core.defineConfig("Fake", "nucleus-channel", "Filter Wheel", "Label", "mScarlet3(569/582)")
core.defineConfig("Real", "membrane-channel", "LED", "Label", "RED")
core.defineConfig("Real", "membrane-channel", "Filter Wheel", "Label", "miRFP670(642/670)")

core.setConfig("Fake", "nucleus-channel")

#print(core.getDeviceInitializationState("LED").__dict__)
#viewer = napari.Viewer()

#viewer.window.add_plugin_dock_widget(plugin_name='napari-micromanager')

#napari.run()

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
layout.addWidget(StageWidget(device="ZStage",mmcore=core))
window.setLayout(layout)
window.setLayout(layout)
window.resize(800, 600)
window.show()
app.exec()