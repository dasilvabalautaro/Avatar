from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.models import AndroidEnvironment
from avatar_face.domain.ports import AndroidDeviceProbe


@dataclass(frozen=True, slots=True)
class InspectAndroidEnvironment:
    """Caso de uso para auditar ADB y los dispositivos conectados."""

    probe: AndroidDeviceProbe

    def execute(self) -> AndroidEnvironment:
        return self.probe.inspect()
