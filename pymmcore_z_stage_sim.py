from pymmcore_plus.experimental.unicore import StageDevice
from microscope_sim import MicroscopeSim
from pymmcore_plus import FocusDirection


class SimZStageDevice(StageDevice):

    def __init__(self, microscope_sim: MicroscopeSim):

        super().__init__()
        self._z_current = 0.0
        self._z_old = 0.0
        self._z_origin = 0.0 # see later
        self._direction = 0 # allowed values (-1,0,1)
        self._microscope_sim = microscope_sim

    def home(self) -> None:
        self._z_current=0.0
        self._z_old=0.0
        self.update_z_camera()

    def stop(self) -> None:
        """
        Stop the movement of the stage
        """
        return

    def set_origin(self) -> None:
        self._z_current = 0.0
        self._z_old = 0.0
        self.update_z_camera()

    def set_position_um(self, val: float) -> None:
        self._z_old = self._z_current
        self._z_current = val
        # update sim
        self.update_z_camera()
        # update focus direction
        self._direction = self._calculate_direction()

    def get_position_um(self) -> float:
        return self._z_current

    def update_z_camera(self):
        self._microscope_sim.set_focal_plane(self._z_current)

    def _calculate_direction(self):
        """Calculate the direction of the stage"""
        return self._z_current - self._z_old

    def get_focus_direction(self) -> FocusDirection:
        """Translate the movement of the z stage"""
        if self._direction > 0.0:
            return FocusDirection.FocusDirectionTowardSample
        else:
            return FocusDirection.FocusDirectionAwayFromSample

    def set_focus_direction(self, sign) -> None:
        """Set the focus direction of the stage
        if sign > 0.0 direction towards sample
        if sign < 0.0 direction away from sample
        """
        # update z pos
        new_position = self._z_current + sign
        self.set_position_um(new_position)





