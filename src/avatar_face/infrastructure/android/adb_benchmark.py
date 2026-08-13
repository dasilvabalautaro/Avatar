from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avatar_face.domain.benchmarking import AndroidBenchmarkRequest, AndroidBenchmarkResult


@dataclass(frozen=True, slots=True)
class AdbBenchmarkRunner:
    """Ejecuta el APK instrumental usando siempre un serial ADB explícito."""

    executable: str = "adb"
    package: str = "com.avatarface.app"
    activity: str = ".MainActivity"
    timeout_seconds: float = 120.0
    poll_interval_seconds: float = 0.25

    def run(self, request: AndroidBenchmarkRequest) -> tuple[AndroidBenchmarkResult, ...]:
        if not request.apk_path.is_file():
            raise FileNotFoundError(f"APK no encontrado: {request.apk_path}")

        self._adb(request.serial, "get-state")
        self._adb(request.serial, "install", "-r", str(request.apk_path))
        request.output_directory.mkdir(parents=True, exist_ok=True)

        results = []
        for backend in request.backends:
            self._adb(request.serial, "shell", "am", "force-stop", self.package)
            self._adb(
                request.serial,
                "shell",
                "run-as",
                self.package,
                "rm",
                "-f",
                "files/benchmark-result.json",
            )
            self._adb(
                request.serial,
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                f"{self.package}/{self.activity}",
                "--es",
                "backend",
                backend,
                "--ei",
                "runs",
                str(request.runs),
                "--es",
                "model",
                request.model_asset,
                "--ez",
                "profile_operators",
                str(request.profile_operators).lower(),
            )
            try:
                payload = self._wait_for_result(request.serial, backend, request.model_asset)
            finally:
                self._adb(
                    request.serial,
                    "shell",
                    "am",
                    "force-stop",
                    self.package,
                    check=False,
                )
            payload["device"] = self._device_metadata(request.serial)
            model_label = Path(request.model_asset).stem.removeprefix("avatarface-feasibility-")
            output_path = request.output_directory / f"{model_label}-{backend}.json"
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            profile_file = payload.get("profile_file")
            if isinstance(profile_file, str):
                profile_result = self._adb(
                    request.serial,
                    "shell",
                    "run-as",
                    self.package,
                    "cat",
                    f"files/{profile_file}",
                )
                profile_path = output_path.with_suffix(".ort-profile.json")
                profile_path.write_text(profile_result.stdout, encoding="utf-8")
                payload["profile_output_path"] = str(profile_path)
                output_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            results.append(AndroidBenchmarkResult(backend, output_path, payload))
        return tuple(results)

    def _wait_for_result(self, serial: str, backend: str, model_asset: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        last_error = "resultado aún no disponible"
        while time.monotonic() < deadline:
            completed = self._adb(
                serial,
                "shell",
                "run-as",
                self.package,
                "cat",
                "files/benchmark-result.json",
                check=False,
            )
            if completed.returncode == 0:
                try:
                    payload: dict[str, Any] = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    last_error = str(error)
                else:
                    reported_backend = payload.get("backend_requested", payload.get("backend"))
                    if reported_backend == backend and payload.get("model") == model_asset:
                        if payload.get("status") != "ok":
                            raise RuntimeError(f"Benchmark Android falló: {payload}")
                        return payload
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Tiempo agotado esperando backend={backend}: {last_error}")

    def _device_metadata(self, serial: str) -> dict[str, str]:
        properties = {
            "manufacturer": "ro.product.manufacturer",
            "model": "ro.product.model",
            "android_version": "ro.build.version.release",
            "sdk": "ro.build.version.sdk",
            "abi": "ro.product.cpu.abi",
            "soc": "ro.soc.model",
        }
        return {
            name: self._adb(serial, "shell", "getprop", prop).stdout.strip()
            for name, prop in properties.items()
        } | {"serial": serial}

    def _adb(
        self,
        serial: str,
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, "-s", serial, *arguments],
            check=check,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
