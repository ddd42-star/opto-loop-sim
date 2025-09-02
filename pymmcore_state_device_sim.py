from pymmcore_plus.experimental.unicore import StateDevice


class SimFilterWheelDevice(StateDevice):

    def __init__(self):
        # add STATES for uv, blue, green, red, farred)
        state_labels = {
            0: "RED",
            1: "BLUE", # excitation 405 nm, emission filter 450/500 nm
            2: "GREEN", # excitation 488 nm, emission filter 525 nm
            3: "UV",
            4: "FARRED"
        }
        super().__init__(state_labels)
        self._current_state = 0 # default position

    def get_state(self) -> int:
        """
        Return the current state of the filter wheel
        """
        return self._current_state

    def set_state(self, position: int) -> None:
        """
        Set the current state of the filter wheel
        """
        if position not in self._state_to_label.keys():
            raise ValueError("Invalid position")
        self._current_state = position


class SimLEDDevice(StateDevice):

    def __init__(self):
        state_labels = {
            0: "RED",
            1: "BLUE",
            2: "GREEN",
            3: "UV",
            4: "FARED"
        }
        super().__init__(state_labels)
        self._current_state = 0

    def get_state(self) -> int:
        """
        Return the current state of the LED Channel
        """
        return self._current_state

    def set_state(self, position: int) -> None:
        """
        Set the current state of the LED Channel
        """
        if position not in self._state_to_label.keys():
            raise ValueError("Invalid position")
        self._current_state = position