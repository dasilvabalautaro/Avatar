from __future__ import annotations

from dataclasses import dataclass


class InvalidPromptError(ValueError):
    """El prompt no cumple el contrato de generación."""


@dataclass(frozen=True, slots=True)
class AvatarPrompt:
    """Solicitud validada de generación de un rostro de avatar."""

    text: str
    seed: int = 42
    image_size: int = 256

    def __post_init__(self) -> None:
        normalized = self.text.strip()
        if not normalized:
            raise InvalidPromptError("El prompt no puede estar vacío.")
        if len(normalized) > 500:
            raise InvalidPromptError("El prompt no puede superar 500 caracteres.")
        if not 0 <= self.seed <= 2**32 - 1:
            raise InvalidPromptError("La seed debe estar entre 0 y 2^32-1.")
        if self.image_size not in {256, 512}:
            raise InvalidPromptError("El tamaño debe ser 256 o 512 píxeles.")
        object.__setattr__(self, "text", normalized)


@dataclass(frozen=True, slots=True)
class AndroidDevice:
    """Dispositivo informado por Android Debug Bridge."""

    serial: str
    state: str
    attributes: tuple[tuple[str, str], ...] = ()

    @property
    def ready(self) -> bool:
        return self.state == "device"

    def attribute(self, name: str) -> str | None:
        return dict(self.attributes).get(name)


@dataclass(frozen=True, slots=True)
class AndroidEnvironment:
    """Resultado neutral de inspeccionar ADB y los dispositivos visibles."""

    adb_path: str | None
    adb_version: str | None
    devices: tuple[AndroidDevice, ...]
    error: str | None = None

    @property
    def ready_devices(self) -> tuple[AndroidDevice, ...]:
        return tuple(device for device in self.devices if device.ready)
