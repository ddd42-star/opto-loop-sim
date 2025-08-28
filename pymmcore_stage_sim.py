from abc import ABC

from pymmcore_plus.experimental.unicore import XYStageDevice

class SimStageDevice(XYStageDevice, ABC):


    def __init__(self):
        super().__init__()
        self._x = 0.0
        self._y = 0.0

    def home(self) -> None:
        """
        Move to its home position
        """
        self._x = 0.0
        self._y = 0.0

    def set_position_um(self, x: float, y: float) -> None:
        """
        Set the stage position using microns
        """
        self._x = x
        self._y = y

    def get_position_um(self) -> tuple[float, float]:
        """
        Return a float representing the current stage position
        """
        return self._x, self._y


    def set_origin_x(self) -> None:
        """
        Set the x coordinate of the stage origin
        """
        self._x = 0.0


    def set_origin_y(self) -> None:
        """
        Set the y coordinate of the stage origin
        """
        self._y = 0.0
