import io

from rich.console import Console

from woodshed import microphones, widgets

BUILTIN, USB = "coreaudio_device_type_builtin", "coreaudio_device_type_usb"
BLUETOOTH, VIRTUAL = "coreaudio_device_type_bluetooth", "coreaudio_device_type_virtual"


def quiet():
    return Console(file=io.StringIO())


def test_pick_moves_with_arrows_and_wraps_around():
    options = ["a", "b", "c"]
    assert widgets.pick(quiet(), options, 0, keys=iter(["down", "down", "up", "enter"])) == 1
    assert widgets.pick(quiet(), options, 0, keys=iter(["up", "enter"])) == 2  # wraps to the bottom
    assert widgets.pick(quiet(), options, 2, keys=iter(["j", "enter"])) == 0  # vim keys too


def test_slide_steps_and_stays_in_range():
    def slide(start, keys):
        return widgets.slide(quiet(), "x", start, 0, 10, 0.5, str, keys=iter(keys))

    assert slide(5, ["right", "right", "left", "enter"]) == 5.5
    assert slide(5, ["up", "enter"]) == 7.5  # ↑ is five steps
    assert slide(9, ["up", "up", "enter"]) == 10  # clamped at the top
    assert slide(1, ["down", "enter"]) == 0  # and at the bottom


def test_slider_preview_follows_the_value():
    console = quiet()
    widgets.slide(console, "x", 5, 0, 10, 1, str, preview=lambda v: f"preview {v:g}", keys=iter(["right", "enter"]))
    assert "x 6" in console.file.getvalue()


def test_suggestion_prefers_what_you_chose_then_interfaces_then_the_mac_microphone():
    names = ["Buds", "Teams Audio", "MacBook Pro Microphone", "Scarlett 2i2"]
    connections = {"Buds": BLUETOOTH, "Teams Audio": VIRTUAL, "MacBook Pro Microphone": BUILTIN, "Scarlett 2i2": USB}
    assert microphones.suggest(names, connections, "Buds", None) == "Buds"
    assert microphones.suggest(names, connections, None, "Buds") == "Scarlett 2i2"
    del connections["Scarlett 2i2"]
    assert microphones.suggest(names[:3], connections, None, "Buds") == "MacBook Pro Microphone"
    assert microphones.suggest(["Buds", "Teams Audio"], connections, None, "Buds") == "Buds"  # nothing better


def test_connections_keep_only_inputs(monkeypatch):
    report = b'''{"SPAudioDataType": [{"_items": [
        {"_name": "Buds", "coreaudio_device_input": 1, "coreaudio_device_transport": "coreaudio_device_type_bluetooth"},
        {"_name": "Speakers", "coreaudio_device_transport": "coreaudio_device_type_builtin"}]}]}'''
    monkeypatch.setattr(microphones.subprocess, "run", lambda *a, **k: type("Done", (), {"stdout": report})())
    assert microphones.connections() == {"Buds": BLUETOOTH}
    assert microphones.describe(BLUETOOTH) == "Bluetooth: phone-call quality"
