


class SimLightDevice:

# illumination on off
# adjusting light intensity
# switch light sources
# to check
    def __init__(self):
        self._on = False
        self._intensity = 0.0  # 0.0 to 1.0
        self._source = "LED1"  # example light source name
        self._available_sources = ["LED1", "LED2", "Laser"]

    def turn_on(self):
        self._on = True

    def turn_off(self):
        self._on = False

    def is_on(self) -> bool:
        return self._on

    def set_intensity(self, intensity: float):
        """Set intensity between 0.0 and 1.0"""
        self._intensity = max(0.0, min(1.0, intensity))

    def get_intensity(self) -> float:
        return self._intensity

    def set_source(self, source: str):
        if source in self._available_sources:
            self._source = source
        else:
            raise ValueError(f"Unknown light source: {source}")

    def get_source(self) -> str:
        return self._source

    def get_available_sources(self) -> list[str]:
        return self._available_sources


