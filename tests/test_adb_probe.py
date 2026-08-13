from avatar_face.infrastructure.android.adb_probe import AdbDeviceProbe


def test_parse_devices_supports_ready_and_unauthorized_devices() -> None:
    output = """List of devices attached
ABC123 device product:pixel model:Pixel_8 device:shiba transport_id:1
XYZ987 unauthorized usb:1-2 transport_id:2
"""

    devices = AdbDeviceProbe._parse_devices(output)

    assert len(devices) == 2
    assert devices[0].serial == "ABC123"
    assert devices[0].attribute("model") == "Pixel_8"
    assert devices[0].ready
    assert not devices[1].ready


def test_parse_devices_ignores_headers_and_daemon_messages() -> None:
    output = """* daemon started successfully
List of devices attached

"""

    assert AdbDeviceProbe._parse_devices(output) == ()
