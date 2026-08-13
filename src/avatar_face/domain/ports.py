from __future__ import annotations

from typing import Protocol

from avatar_face.domain.models import AndroidEnvironment


class AndroidDeviceProbe(Protocol):
    """Puerto para inspeccionar Android sin acoplar la aplicación a ADB."""

    def inspect(self) -> AndroidEnvironment:
        """Devuelve el estado observable del entorno Android."""
        ...
