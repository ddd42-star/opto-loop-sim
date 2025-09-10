from pymmcore_plus.experimental.unicore import StateDevice


class SimStateDevice(StateDevice):

    def __init__(self, state_dict: dict[int, str]):
        super().__init__(state_dict)
        self._current_state = 0 # default position

    def get_state(self) -> int:
        """
        Return the current state of the filter wheel
        """
        return self._current_state

    def set_state(self, position: int | str) -> None:
        """
        Set the current state of the filter wheel
        """
        if isinstance(position, str):
            position = int(position)

        self._current_state = position