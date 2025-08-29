from pymmcore_plus.experimental.unicore import StateDevice
from typing import Mapping, Iterable

class SimStateDevice(StateDevice):


    def __init__(self, state_labels: Mapping[int, str] | Iterable[tuple[int, str]]) -> None:
        super().__init__(state_labels)


    def get_state(self) -> int:
        pass

    def set_state(self, state: int) -> None:
        pass

