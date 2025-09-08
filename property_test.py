from napari_micromanager._gui_objects._toolbar import MicroManagerToolbar
from pymmcore_plus.experimental.unicore import UniMMCore
import napari
from pymmcore_widgets import PropertiesWidget
from pymmcore_plus import CMMCorePlus, PropertyType
from qtpy.QtWidgets import QWidget
from qtpy.QtWidgets import QSizePolicy
from napari_micromanager._gui_objects._illumination_widget import IlluminationWidget


class MyMicromanagerToolbar(MicroManagerToolbar):

    def __init__(self, viewer: napari.viewer.Viewer, core: UniMMCore):
        super().__init__(viewer=viewer)
        self._mmc = core



class MyIllumination(PropertiesWidget):

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        mmcore: UniMMCore | None = None,
    ):
        print(mmcore)
        super().__init__(
            property_name_pattern="(Intensity|Power|Level|Brightness|test)s?",
            property_type={PropertyType.Integer, PropertyType.Float},
            parent=parent,
            mmcore=mmcore,
        )

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


