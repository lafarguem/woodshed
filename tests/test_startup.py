import subprocess
import sys


def test_commands_start_without_loading_the_audio_libraries():
    # torch alone took half a second to import, before every command: even `shed --help` and `shed songs`.
    code = ("import sys, woodshed.cli; "
            "print(sorted({'torch', 'librosa', 'demucs', 'mlx_whisper', 'sounddevice'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
