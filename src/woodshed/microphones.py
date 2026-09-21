"""How each microphone or audio interface is connected, and which one to suggest."""

import json
import subprocess

_BLUETOOTH, _VIRTUAL = "coreaudio_device_type_bluetooth", "coreaudio_device_type_virtual"
_DESCRIPTIONS = {
    "coreaudio_device_type_builtin": "built-in",
    "coreaudio_device_type_usb": "USB audio interface",
    "coreaudio_device_type_thunderbolt": "Thunderbolt audio interface",
    _BLUETOOTH: "Bluetooth: phone-call quality",
    _VIRTUAL: "virtual: app audio, not a microphone",
    "coreaudio_device_type_aggregate": "aggregate device",
}
# Best first: an audio interface, then the Mac's own microphone.
_PREFERRED = ("coreaudio_device_type_usb", "coreaudio_device_type_thunderbolt", "coreaudio_device_type_builtin")


def connections() -> dict[str, str]:
    """Input device name -> how macOS says it's connected. Empty if macOS won't say."""
    try:
        out = subprocess.run(["system_profiler", "SPAudioDataType", "-json"],
                             capture_output=True, timeout=10, check=True).stdout
        items = json.loads(out)["SPAudioDataType"][0]["_items"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return {}
    return {item["_name"]: item.get("coreaudio_device_transport", "") for item in items
            if item.get("coreaudio_device_input")}


def describe(connection: str) -> str:
    return _DESCRIPTIONS.get(connection, "")


def suggest(names: list[str], connections: dict[str, str], current: str | None, system_default: str | None) -> str:
    """The one you chose before, else an audio interface, else the built-in mic, never Bluetooth if avoidable."""
    if current in names:
        return current
    for kind in _PREFERRED:
        if found := [name for name in names if connections.get(name) == kind]:
            return found[0]
    if not connections:  # macOS didn't say: go by name
        if found := [name for name in names if any(k in name for k in ("MacBook", "iMac", "Built-in"))]:
            return found[0]
    usable = [name for name in names if connections.get(name) not in (_BLUETOOTH, _VIRTUAL)]
    if system_default in usable:
        return system_default
    return (usable or names)[0]
