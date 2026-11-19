"""AudioContainer, LSB, FileCarver, MetadataScanner, SafeArchiveInspector ve hash testleri."""
import hashlib
import io
import os
import zipfile

import numpy as np
import pytest

import synth
from core.audio_loader import AudioContainer, AudioLoadError
from core.crypto_hasher import calculate_file_hashes
from core.safety import SafeArchiveInspector, SecurityAnomaly
from detectors.file_carver import FileCarver
from detectors.lsb_analyzer import LSBAnalyzer
from detectors.metadata_scanner import MetadataScanner

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _zip(files, method=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


# ---------------- AudioContainer ----------------

@pytest.mark.parametrize("sampwidth", [1, 2, 3, 4])
def test_loader_pcm_bit_depths(tmp_path, sampwidth):
    sig = synth.tone(440, 0.5, 0.5)
    path = synth.write_wav(tmp_path / f"t{sampwidth}.wav", sig, sampwidth=sampwidth)
    c = AudioContainer(path)
    mono = c.channels["Mono"]
    assert c.sampwidth == sampwidth
    assert c.duration == pytest.approx(0.5, abs=1e-3)
    assert np.max(np.abs(mono - sig)) < (0.02 if sampwidth == 1 else 1e-3)


def test_loader_stereo_derived_channels(tmp_path):
    left, right = synth.tone(440, 0.3, 0.4), synth.tone(440, 0.3, 0.1)
    inter = np.empty(len(left) * 2, dtype=np.float32)
    inter[0::2], inter[1::2] = left, right
    c = AudioContainer(synth.write_wav(tmp_path / "st.wav", inter, channels=2))
    assert set(c.channels) == {"Channel_Left", "Channel_Right", "Channel_Diff_LR", "Channel_Mono"}
    np.testing.assert_allclose(c.channels["Channel_Diff_LR"], left - right, atol=1e-3)
    np.testing.assert_allclose(c.channels["Channel_Mono"], (left + right) / 2, atol=1e-3)


@pytest.mark.parametrize("bits, dtype", [(32, "<f4"), (64, "<f8")])
def test_loader_ieee_float(tmp_path, bits, dtype):
    sig = synth.tone(440, 0.2, 0.7)
    raw = synth.build_riff([(b"data", sig.astype(dtype).tobytes())], fmt_tag=3, bits=bits)
    path = tmp_path / "f.wav"
    path.write_bytes(raw)
    c = AudioContainer(path)
    np.testing.assert_allclose(c.channels["Mono"], sig, atol=1e-6)


def test_loader_wave_format_extensible_float(tmp_path):
    sig = synth.tone(1000, 0.1, 0.25)
    raw = synth.build_riff([(b"data", sig.astype("<f4").tobytes())], bits=32, extensible_subformat=3)
    path = tmp_path / "ext.wav"
    path.write_bytes(raw)
    np.testing.assert_allclose(AudioContainer(path).channels["Mono"], sig, atol=1e-6)


def test_loader_float_nan_is_sanitised(tmp_path):
    data = np.array([0.1, np.nan, np.inf, -np.inf] * 20, dtype="<f4")
    path = tmp_path / "nan.wav"
    path.write_bytes(synth.build_riff([(b"data", data.tobytes())], fmt_tag=3, bits=32))
    assert np.isfinite(AudioContainer(path).channels["Mono"]).all()


def test_loader_truncated_odd_data_chunk(tmp_path):
    raw = synth.build_riff([(b"data", b"\x01\x00" * 100)])
    path = tmp_path / "trunc.wav"
    path.write_bytes(raw[:-1])  # cut half a sample
    c = AudioContainer(path)
    assert len(c.channels["Mono"]) == 99


def test_loader_appended_payload(tmp_path):
    path = synth.write_wav(tmp_path / "a.wav", synth.tone(440, 0.1))
    with open(path, "ab") as f:
        f.write(b"TRAILING-SECRET")
    assert AudioContainer(path).appended_payload == b"TRAILING-SECRET"


def test_loader_list_chunk_after_data_is_metadata_not_slack(tmp_path):
    # Many editors write LIST/INFO *after* the data chunk; that is not hidden slack
    raw = synth.build_riff([(b"data", b"\x00\x00" * 50), synth.info_list_chunk({"INAM": "title"})])
    path = tmp_path / "late_list.wav"
    path.write_bytes(raw)
    c = AudioContainer(path)
    assert c.appended_payload == b""
    assert [m["id"] for m in c.metadata_chunks] == ["LIST"]


def test_loader_appended_zip_is_not_parsed_as_chunks(tmp_path):
    zip_bytes = _zip({"s.txt": b"secret"})
    path = tmp_path / "z.wav"
    path.write_bytes(synth.build_riff([(b"data", b"\x00\x00" * 50)]) + zip_bytes)
    c = AudioContainer(path)
    assert c.appended_payload == zip_bytes
    assert c.metadata_chunks == []


def test_loader_undersized_riff_header_does_not_eat_audio(tmp_path):
    path = tmp_path / "small_riff.wav"
    path.write_bytes(synth.build_riff([(b"data", b"\x01\x00" * 200)], riff_size=60))
    c = AudioContainer(path)
    assert len(c.channels["Mono"]) == 200 and c.appended_payload == b""


@pytest.mark.parametrize("raw", [b"", b"RIFF", b"NOTAWAVEFILE" * 10, b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 40])
def test_loader_rejects_invalid(tmp_path, raw):
    path = tmp_path / "bad.wav"
    path.write_bytes(raw)
    with pytest.raises(AudioLoadError):
        AudioContainer(path)


def test_loader_hashes_match_hashlib(tmp_path):
    path = synth.write_wav(tmp_path / "h.wav", synth.tone(440, 0.1))
    data = path.read_bytes()
    assert calculate_file_hashes(path) == {
        "md5": hashlib.md5(data).hexdigest(),
        "sha1": hashlib.sha1(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# ---------------- LSB ----------------

@pytest.mark.parametrize("bit", [0, 1, 2])
def test_lsb_embedded_flag_each_bit_plane(tmp_path, bit):
    stego = synth.embed_lsb(synth.noisy_cover(20000), b"flag{lsb_plane_%d}" % bit, bit=bit)
    path = synth.write_int16_wav(tmp_path / "lsb.wav", stego)
    results = LSBAnalyzer(str(path), output_dir=str(tmp_path / "out")).analyze()
    by_bit = {r["bit_index"]: r for r in results}
    assert by_bit[bit]["has_flag"]
    assert f"flag{{lsb_plane_{bit}}}" in [m["decoded"] for m in by_bit[bit]["decoded_messages"]]


def test_lsb_stereo_left_channel_only(tmp_path):
    left = synth.embed_lsb(synth.noisy_cover(20000, seed=1), b"flag{left_only}")
    right = synth.noisy_cover(20000, seed=2)
    inter = np.empty(40000, dtype=np.int16)
    inter[0::2], inter[1::2] = left, right
    path = synth.write_int16_wav(tmp_path / "st.wav", inter, channels=2)
    results = LSBAnalyzer(str(path), output_dir=str(tmp_path / "out")).analyze()
    msgs = [m for r in results for m in r["decoded_messages"]]
    assert any(m["decoded"] == "flag{left_only}" and "Left Channel" in m["variant"] for m in msgs)


def test_lsb_hidden_file_signature(tmp_path):
    stego = synth.embed_lsb(synth.noisy_cover(20000), b"junk" + PNG_HEADER)
    path = synth.write_int16_wav(tmp_path / "png.wav", stego)
    results = LSBAnalyzer(str(path), output_dir=str(tmp_path / "out")).analyze()
    sigs = results[0]["embedded_signatures"]
    assert {"offset": 4, "type": "PNG Image", "bitorder": "big"} in sigs


def test_lsb_random_bits_raise_entropy_and_bitplane_image(tmp_path):
    rng = np.random.default_rng(7)
    cover = synth.noisy_cover(200000)
    stego = synth.embed_lsb(cover, rng.integers(0, 256, 200000 // 8, dtype=np.uint8).tobytes())
    out = tmp_path / "out"
    results = LSBAnalyzer(str(synth.write_int16_wav(tmp_path / "rnd.wav", stego)), output_dir=str(out)).analyze()
    assert results[0]["is_suspicious_entropy"]
    assert (out / "lsb_bitplane_bit0.png").exists()


def test_lsb_clean_silence_has_no_findings(tmp_path):
    path = synth.write_int16_wav(tmp_path / "z.wav", np.zeros(20000, dtype=np.int16))
    results = LSBAnalyzer(str(path), output_dir=str(tmp_path / "out")).analyze()
    assert all(not r["has_flag"] and not r["embedded_signatures"] for r in results)
    assert all(r["entropy"] == 0.0 for r in results)


def test_lsb_24bit(tmp_path):
    sig = synth.tone(440, 0.5, 0.5)
    path = synth.write_wav(tmp_path / "24.wav", sig, sampwidth=3)
    results = LSBAnalyzer(str(path), output_dir=str(tmp_path / "out")).analyze()
    assert results and results[0]["sample_count"] == len(sig)


@pytest.mark.parametrize("data, expected", [
    (np.zeros(1000, dtype=np.uint8), 0.0),
    (np.arange(256, dtype=np.uint8), 8.0),
    (np.array([0, 1] * 500, dtype=np.uint8), 1.0),
    (np.array([], dtype=np.uint8), 0.0),
])
def test_shannon_entropy(data, expected):
    assert LSBAnalyzer.calculate_shannon_entropy(data) == pytest.approx(expected)


# ---------------- FileCarver ----------------

def test_find_signatures_at_offsets():
    blob = b"xx" + PNG_HEADER + b"yy" + b"%PDF-1.7" + b"zz" + b"PK\x03\x04"
    found = FileCarver.find_signatures(blob)
    assert [(o, t) for o, t, _ in found] == [(2, "PNG Image"), (36, "PDF Document"), (46, "ZIP Archive")]


def test_find_signatures_min_length_filter():
    assert FileCarver.find_signatures(b"..\x1f\x8b\x08..", min_magic_len=4) == []
    assert FileCarver.find_signatures(b"..\x1f\x8b\x08..")[0][1] == "GZIP File"


def test_find_signatures_respects_limit():
    assert len(FileCarver.find_signatures(b"GIF89a" * 100, limit=5)) == 5


@pytest.mark.parametrize("magic, mtype, ext", [
    (PNG_HEADER, "PNG Image", ".png"),
    (b"%PDF-1.4 hello world doc", "PDF Document", ".pdf"),
    (b"GIF89a" + b"\x00" * 20, "GIF Image", ".gif"),
    (b"\x7fELF" + b"\x00" * 20, "ELF Executable", ".elf"),
])
def test_carver_identifies_payload_type(tmp_path, magic, mtype, ext):
    res = FileCarver("x.wav", appended_payload=magic, output_dir=str(tmp_path)).scan_and_carve()
    assert res[0]["type"] == mtype
    assert res[0]["extracted_path"].endswith(ext)
    assert open(res[0]["extracted_path"], "rb").read() == magic


def test_carver_finds_zip_behind_junk_padding(tmp_path):
    payload = b"\x00JUNKPADDING" * 4 + _zip({"note.txt": b"flag{behind_junk}"})
    res = FileCarver("x.wav", appended_payload=payload, output_dir=str(tmp_path)).scan_and_carve()
    zips = [r for r in res if r["type"] == "ZIP Archive"]
    assert zips and zips[0]["offset"] == 48
    assert "flag{behind_junk}" in [m["decoded"] for m in zips[0]["decoded_messages"]]


def test_carver_plaintext_flag_in_slack(tmp_path):
    payload = b"\x00\x01\x02 comment: " + b"flag{plain_slack}".hex().encode()
    res = FileCarver("x.wav", appended_payload=payload, output_dir=str(tmp_path)).scan_and_carve()
    assert "flag{plain_slack}" in [m["decoded"] for m in res[0]["decoded_messages"]]


def test_carver_ignores_tiny_payload(tmp_path):
    assert FileCarver("x.wav", appended_payload=b"tiny", output_dir=str(tmp_path)).scan_and_carve() == []


def test_carver_blocks_zip_bomb_in_slack(tmp_path):
    bomb = _zip({"z.bin": b"\0" * (30 * 1024 * 1024)})
    res = FileCarver("x.wav", appended_payload=bomb, output_dir=str(tmp_path)).scan_and_carve()
    assert res[0]["security_alert"] and res[0]["safety_status"].startswith("BLOCKED")


# ---------------- MetadataScanner ----------------

def _container(tmp_path, chunks, riff_size=None, extra=b""):
    raw = synth.build_riff(list(chunks) + [(b"data", b"\x00\x00" * 200)], riff_size=riff_size)
    path = tmp_path / "meta.wav"
    path.write_bytes(raw + extra)
    return AudioContainer(path)


def test_metadata_flag_in_info_comment(tmp_path):
    c = _container(tmp_path, [synth.info_list_chunk({"INAM": "Song", "ICMT": "flag{in_the_comment}"})])
    rep = MetadataScanner.scan(c)
    assert "flag{in_the_comment}" in [f["decoded"] for f in rep["findings"]]
    assert rep["findings"][0]["chunk"] == "LIST"
    assert not rep["anomalies"]


def test_metadata_base64_flag_in_artist(tmp_path):
    import base64
    c = _container(tmp_path, [synth.info_list_chunk({"IART": base64.b64encode(b"flag{artist_b64}").decode()})])
    assert "flag{artist_b64}" in [f["decoded"] for f in MetadataScanner.scan(c)["findings"]]


def test_metadata_nonstandard_chunk_and_embedded_file(tmp_path):
    c = _container(tmp_path, [(b"hide", b"xx" + PNG_HEADER)])
    types = [a["type"] for a in MetadataScanner.scan(c)["anomalies"]]
    assert "Non-Standard Chunk" in types and "Embedded File In Chunk" in types


def test_metadata_riff_size_mismatch(tmp_path):
    c = _container(tmp_path, [], riff_size=100)
    assert any(a["type"] == "RIFF Size Mismatch" for a in MetadataScanner.scan(c)["anomalies"])


def test_metadata_clean_file(tmp_path):
    path = synth.write_wav(tmp_path / "clean.wav", synth.tone(440, 0.1))
    rep = MetadataScanner.scan(AudioContainer(path))
    assert rep["findings"] == [] and rep["anomalies"] == []


# ---------------- SafeArchiveInspector ----------------

def test_safe_zip_extracts_normal_files(tmp_path):
    rep = SafeArchiveInspector.inspect_and_extract_zip(_zip({"a.txt": b"hello", "b.txt": b"world"}), str(tmp_path))
    assert rep["is_safe"] and not rep["anomaly_detected"]
    assert sorted(f["filename"] for f in rep["extracted_files"]) == ["a.txt", "b.txt"]


def test_safe_zip_file_count_limit(tmp_path):
    many = _zip({f"f{i}.txt": b"x" for i in range(SafeArchiveInspector.MAX_FILE_COUNT + 1)})
    rep = SafeArchiveInspector.inspect_and_extract_zip(many, str(tmp_path))
    assert not rep["is_safe"] and "file count" in rep["reason"]


def test_safe_zip_total_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(SafeArchiveInspector, "MAX_UNCOMPRESSED_BYTES", 1000)
    rep = SafeArchiveInspector.inspect_and_extract_zip(_zip({"a.bin": os.urandom(2000)}, zipfile.ZIP_STORED), str(tmp_path))
    assert not rep["is_safe"] and "too large" in rep["reason"]


def test_safe_zip_corrupted(tmp_path):
    rep = SafeArchiveInspector.inspect_and_extract_zip(b"PK\x03\x04garbage", str(tmp_path))
    assert not rep["is_safe"] and "Corrupted" in rep["reason"]


def test_safe_zip_traversal_entry_stays_inside(tmp_path):
    rep = SafeArchiveInspector.inspect_and_extract_zip(_zip({"../../evil.txt": b"x", "/abs/x.txt": b"y"}), str(tmp_path / "o"))
    for f in rep["extracted_files"]:
        assert os.path.dirname(f["path"]) == str(tmp_path / "o")
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("name", ["..", "../", "..\\..\\", "", "a/../../b.txt", "C:\\Windows\\x.dll"])
def test_sanitize_path_variants(tmp_path, name):
    dest = SafeArchiveInspector.sanitize_path(str(tmp_path), name)
    assert os.path.dirname(dest) == str(tmp_path)


def test_recursion_depth_at_limit_is_allowed(tmp_path):
    rep = SafeArchiveInspector.inspect_and_extract_zip(_zip({"a.txt": b"x"}), str(tmp_path), depth=SafeArchiveInspector.MAX_RECURSION_DEPTH)
    assert rep["is_safe"]
    with pytest.raises(SecurityAnomaly):
        SafeArchiveInspector.inspect_and_extract_zip(_zip({"a.txt": b"x"}), str(tmp_path), depth=99)
