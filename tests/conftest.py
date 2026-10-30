import sys
import types
from pathlib import Path

# Make the repo root importable so tests can import app.utils.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# app.utils pulls in app.config (imports torch) and faster_whisper, both of
# which are heavyweight runtime deps irrelevant to load_audio. Stub them so
# the unit tests run in a minimal environment.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

if "faster_whisper" not in sys.modules:
    fw_stub = types.ModuleType("faster_whisper")
    fw_utils_stub = types.ModuleType("faster_whisper.utils")
    fw_utils_stub.format_timestamp = lambda *args, **kwargs: ""
    fw_stub.utils = fw_utils_stub
    sys.modules["faster_whisper"] = fw_stub
    sys.modules["faster_whisper.utils"] = fw_utils_stub
