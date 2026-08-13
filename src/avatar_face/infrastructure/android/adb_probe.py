from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from avatar_face.domain.models import AndroidDevice, AndroidEnvironment


@dataclass(frozen=True, slots=True)
class AdbDeviceProbe:
    """Inspecciona dispositivos mediante el ejecutable ADB local."""

    executable: str = "adb"
    timeout_seconds: float = 10.0

    def inspect(self) -> AndroidEnvironment:
        path = shutil.which(self.executable)
        if path is None:
            return AndroidEnvironment(None, None, (), "ADB no está disponible en PATH.")

        try:
            version = self._run(path, "version").stdout.splitlines()[0]
            devices_output = self._run(path, "devices", "-l").stdout
        except (OSError, subprocess.SubprocessError) as error:
            return AndroidEnvironment(path, None, (), str(error))

        return AndroidEnvironment(path, version, self._parse_devices(devices_output))

    def _run(self, path: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [path, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )

    @staticmethod
    def _parse_devices(output: str) -> tuple[AndroidDevice, ...]:
        devices: list[AndroidDevice] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("List of devices") or line.startswith("*"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            attributes: list[tuple[str, str]] = []
            for field in fields[2:]:
                if ":" not in field:
                    continue
                name, value = field.split(":", maxsplit=1)
                attributes.append((name, value))
            devices.append(AndroidDevice(fields[0], fields[1], tuple(attributes)))
        return tuple(devices)
