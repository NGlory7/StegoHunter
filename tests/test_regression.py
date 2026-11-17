"""
Regresyon testleri: README'deki CTF çözümleri ve güvenlik korumaları.
Örnek dosyalar samples/ klasöründe (kaynakları samples/README.md'de).
"""
import base64
import io
import os
import sys
import tempfile
import zipfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.audio_loader import AudioContainer
from core.safety import SafeArchiveInspector, SecurityAnomaly
from decoders.text_decoders import TextDecoderHub
from detectors.file_carver import FileCarver
from detectors.lsb_analyzer import LSBAnalyzer
from detectors.signal_decoders import SignalDecoders

SAMPLES = os.path.join(ROOT, "samples")


def _signal(name):
    c = AudioContainer(os.path.join(SAMPLES, name))
    return next(iter(c.channels.values())), c.sample_rate


# ---------------- CTF örnekleri ----------------

def test_dtmf_beep_boop():
    sig, sr = _signal("flag.wav")
    res = SignalDecoders.detect_dtmf(sig, sr)
    assert "flag{do_you_speak_the_beep_boop}" in [m["decoded"] for m in res["messages"]]


def test_pam_surfing_waves():
    sig, _ = _signal("picoctf_surfing_waves.wav")
    res = SignalDecoders.detect_pam_discrete_levels(sig)
    assert "picoCTF{mU21C_1s_1337_115155af}" in [m["decoded"] for m in res["messages"]]


def test_lsb_walkman():
    with tempfile.TemporaryDirectory() as tmp:
        results = LSBAnalyzer(os.path.join(SAMPLES, "wazaaa.wav"), output_dir=tmp).analyze()
    decoded = [m["decoded"] for r in results for m in r.get("decoded_messages", [])]
    assert "flag{do_that_bit_again}" in decoded


def test_clean_samples_have_no_false_flags():
    # DTMF / PAM dedektörleri ilgisiz dosyalarda bayrak uydurmamalı
    sig, sr = _signal("sample_challenge.wav")
    assert not (SignalDecoders.detect_dtmf(sig, sr) or {}).get("messages")
    assert not (SignalDecoders.detect_pam_discrete_levels(sig) or {}).get("messages")


# ---------------- Metin çözücüler ----------------

def test_text_decoders():
    assert TextDecoderHub.decode_base64_token(base64.b64encode(b"flag{b64}").decode()) == "flag{b64}"
    assert TextDecoderHub.decode_hex_token(b"flag{hex}".hex()) == "flag{hex}"
    assert TextDecoderHub.decode_rot13("synt{ebg}") == "flag{rot}"
    assert TextDecoderHub.decode_morse("... --- ...") == "SOS"


# ---------------- Güvenlik korumaları ----------------

def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_zip_bomb_ratio_is_blocked():
    bomb = _zip({"zeros.bin": b"\0" * (20 * 1024 * 1024)})  # ~1000:1 sıkıştırma
    with tempfile.TemporaryDirectory() as tmp:
        report = SafeArchiveInspector.inspect_and_extract_zip(bomb, tmp)
        assert not report["is_safe"]
        assert not os.listdir(tmp), "şüpheli arşiv diske açılmamalı"


def test_path_traversal_is_neutralised():
    with tempfile.TemporaryDirectory() as tmp:
        dest = SafeArchiveInspector.sanitize_path(tmp, "../../evil.bat")
        assert os.path.dirname(dest) == os.path.abspath(tmp)


def test_recursion_depth_limit():
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(SecurityAnomaly):
        SafeArchiveInspector.inspect_and_extract_zip(_zip({"a.txt": b"x"}), tmp, depth=SafeArchiveInspector.MAX_RECURSION_DEPTH + 1)


def test_appended_zip_payload_is_carved():
    from demo_generator import generate_multi_stego_demo

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "demo.wav")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            generate_multi_stego_demo(filename=wav, message="FLAG{TEST}", duration=2.0)
        finally:
            os.chdir(cwd)
        container = AudioContainer(wav)
        assert container.appended_payload, "demo dosyasının sonuna ZIP eklenmiş olmalı"
        carved = FileCarver(wav, appended_payload=container.appended_payload, output_dir=os.path.join(tmp, "out")).scan_and_carve()
        assert carved and not any(c.get("security_alert") for c in carved)
