"""Spektrogram pankart OCR'ı: normalleştirme, oylama, satır genişletme ve uçtan uca okuma."""
import os

import cv2
import numpy as np
import pytest

import synth
import detectors.ocr_engine as ocr_mod
from detectors.ocr_engine import (OCREngine, TESSERACT_AVAILABLE, consensus_body, consensus_prefix,
                                  normalize_flag_text, weighted_medoid)
from detectors.sliding_spectrogram import SlidingSpectrogramScanner

needs_tesseract = pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed")
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


# ---------------- Normalleştirme (Tesseract gerekmez) ----------------

@pytest.mark.parametrize("raw, expected", [
    ("TE(Music is Fr33d0m)}", ("TE", "Music_is_Fr33d0m")),
    ("HRCTE(Music is Fr33d0m}", ("HRCTF", "Music_is_Fr33d0m")),       # CTE( -> CTF{
    ("cheeCT FiMusic is Fe33d0m;", ("cheeCTF", "Music_is_Fe33d0m")),  # 'CT Fi' -> CTF{
    ("flag(ocr spectra)", ("flag", "ocr_spectra")),                  # case preserved
    ("flagiocr spectro}", ("flag", "ocr_spectro")),                  # 'i' read for '{'
    ("TE{Music . is Fr33d0m}", ("TE", "Music_is_Fr33d0m")),          # speckle token dropped
    ("x [abc def]", ("x", "abc_def")),
])
def test_normalize_flag_text(raw, expected):
    assert normalize_flag_text(raw) == expected


@pytest.mark.parametrize("raw", ["", "hello world", "TE{ab}", "{ . , }", "no braces here at all"])
def test_normalize_flag_text_rejects(raw):
    assert normalize_flag_text(raw) is None


def test_weighted_medoid_prefers_central_reading():
    votes = {"Music_is_Fr33d0m": 1.0, "Music_is_Fr33dGm": 1.0, "Music_is_Fe33d0m": 1.0, "zzzz": 1.5}
    assert weighted_medoid(votes) == "Music_is_Fr33d0m"


def test_consensus_body_ignores_random_word_splits():
    votes = {"Music_is_Fr33d0m": 1.5, "Music_is_Fr3_3d0m": 1.2, "Music_is_Fr33dGm": 1.0}
    assert consensus_body(votes) == "Music_is_Fr33d0m"


def test_consensus_body_keeps_majority_underscores():
    votes = {"ocr_spectro": 2.0, "ocrspectro": 1.0, "ocr_spectro_": 1.0}
    assert consensus_body(votes) == "ocr_spectro"


def test_consensus_prefix_stops_where_readings_disagree():
    votes = {"HRCTF": 1.5, "nftCTF": 1.5, "PRCTF": 1.4, "cheeCTF": 1.3, "RCTF": 1.6}
    assert consensus_prefix(votes) == "CTF"


def test_consensus_prefix_case_insensitive():
    assert consensus_prefix([("flag", 1.5), ("FLAG", 1.0), ("Flag", 1.2)]).lower() == "flag"
    assert consensus_prefix([("flag", 1.0)] * 4) == "flag"  # identical readings still count


def test_consensus_prefix_needs_support():
    assert consensus_prefix({"CTF": 1.0}) == ""
    assert consensus_prefix({}) == ""


def test_find_tesseract_honours_env(tmp_path, monkeypatch):
    fake = tmp_path / "tesseract.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    assert ocr_mod._find_tesseract() == str(fake)


def test_read_banner_without_tesseract_is_graceful(monkeypatch):
    monkeypatch.setattr(ocr_mod, "TESSERACT_AVAILABLE", False)
    res = OCREngine().read_banner(np.full((60, 400), 255, np.uint8))
    assert res["available"] is False and res["decoded_candidates"] == []


# ---------------- Satır genişletme ----------------

def test_expand_to_text_line_joins_words():
    thresh = np.zeros((100, 600), np.uint8)
    for x0, x1 in [(20, 120), (150, 330), (360, 560)]:   # three words on one line
        cv2.rectangle(thresh, (x0, 40), (x1, 70), 255, -1)
    cv2.rectangle(thresh, (200, 0), (220, 8), 255, -1)  # noise above the line
    x, y, w, h = SlidingSpectrogramScanner._expand_to_text_line((150, 40, 181, 31), thresh)
    assert (x, x + w) == (20, 561) and y == 40 and h == 31


def test_expand_to_text_line_does_not_jump_big_gaps():
    thresh = np.zeros((100, 900), np.uint8)
    cv2.rectangle(thresh, (20, 40), (200, 70), 255, -1)
    cv2.rectangle(thresh, (700, 40), (800, 70), 255, -1)  # gap of 500 px >> 1.6 x line height
    x, _, w, _ = SlidingSpectrogramScanner._expand_to_text_line((20, 40, 181, 31), thresh)
    assert (x, x + w) == (20, 201)


# ---------------- Tesseract ile okuma ----------------

def _banner(text, noise=0.0, seed=0):
    img = np.zeros((80, 40 + 22 * len(text)), np.uint8)
    cv2.putText(img, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 230, 4, cv2.LINE_AA)  # spectrogram strokes are thick
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0]), interpolation=cv2.INTER_LINEAR)  # wide letters
    if noise:
        img = np.clip(img + np.random.default_rng(seed).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


@needs_tesseract
@pytest.mark.parametrize("noise", [0.0, 20.0])
def test_read_banner_clean_and_noisy(noise):
    res = OCREngine().read_banner(_banner("flag{read_me}", noise=noise))
    assert "flag{read_me}" in [c["decoded"] for c in res["decoded_candidates"]]
    assert all(c["confidence"] != "VERY HIGH" for c in res["decoded_candidates"] if "ocr_votes" in c)


@needs_tesseract
def test_read_banner_pure_noise_reports_nothing():
    noise = np.random.default_rng(1).integers(0, 255, (80, 900)).astype(np.uint8)
    assert OCREngine().read_banner(noise)["decoded_candidates"] == []


@needs_tesseract
def test_scanner_extends_window_and_reads_flag(tmp_path):
    sig = synth.spectrogram_text_signal("flag{ocr_spectro}")  # text spans 1-9 s, windows are 3.5 s
    sc = SlidingSpectrogramScanner({"Mono": sig}, 22050, len(sig) / 22050, output_dir=str(tmp_path), window_sec=3.5, overlap=0.5)
    _, _, top = sc.scan()
    decoded = [c["decoded"] for h in top for c in h["decoded_candidates"]]
    assert "flag{ocr_spectro}" in decoded
    assert any(h.get("banner_extended") for h in top)
    assert all(os.path.exists(tmp_path / h["banner_file"]) for h in top if h.get("banner_file"))


def test_refine_widens_cut_banner_without_tesseract(tmp_path, monkeypatch):
    # A window in the middle of an 8 s text banner: the banner touches both edges
    monkeypatch.setattr(ocr_mod, "TESSERACT_AVAILABLE", False)
    sig = synth.spectrogram_text_signal("flag{no_ocr_needed}", duration=10.0, t0=1.0, t1=9.0)
    sc = SlidingSpectrogramScanner({"Mono": sig}, 22050, 10.0, output_dir=str(tmp_path), window_sec=3.5, overlap=0.5)
    sc.global_peak_mag = float(np.max(np.abs(sig)))
    h = {"channel": "Mono", "window_label": "Window_03", "time_start": 3.5, "time_end": 7.0,
         "time_str": "3.50s - 7.00s", "n_fft": 512, "f_min": 3858, "f_max": 8268, "decoded_candidates": []}
    sc._refine_highlight(h, 0, set())
    assert h["banner_extended"] and h["original_time_str"] == "3.50s - 7.00s"
    assert h["time_start"] < 3.5 and h["time_end"] > 7.0
    assert os.path.exists(tmp_path / h["banner_file"])
    assert h["banner_ocr"]["available"] is False and h["decoded_candidates"] == []


@needs_tesseract
def test_riftctf_banner_is_read_by_ocr(tmp_path):
    from core.audio_loader import AudioContainer

    c = AudioContainer(os.path.join(SAMPLES, "riftctf_chall3.wav"))
    sr = c.sample_rate
    left = c.channels["Channel_Left"][3 * sr: 14 * sr]  # flag is drawn at ~5-10.5 s
    sc = SlidingSpectrogramScanner({"Channel_Left": left}, sr, len(left) / sr, output_dir=str(tmp_path), window_sec=3.5, overlap=0.5)
    _, _, top = sc.scan()
    bodies = [c["decoded"].split("{", 1)[-1] for h in top for c in h["decoded_candidates"] if "ocr_votes" in c]
    assert "Music_is_Fr33d0m}" in bodies


def test_text_column_span_bridges_underscores():
    img = np.zeros((120, 1000), np.uint8)
    for x0, x1 in [(50, 250), (420, 620), (790, 950)]:     # three words
        img[40:80, x0:x1] = 220
    img[76:80, 250:420] = 220                               # '_' joining word 1 and 2
    img[76:80, 620:790] = 220                               # '_' joining word 2 and 3
    a, b = SlidingSpectrogramScanner._text_column_span(img, (420, 40, 200, 40))
    assert a <= 60 and b >= 940


def test_text_column_span_ignores_uniform_texture():
    rng = np.random.default_rng(4)
    img = rng.integers(60, 200, (120, 1000)).astype(np.uint8)  # music-like texture everywhere
    img[40:80, 400:600] = 250                                   # bright text block
    a, b = SlidingSpectrogramScanner._text_column_span(img, (400, 40, 200, 40))
    assert 350 <= a <= 420 and 580 <= b <= 650


def test_demo_generator_keeps_long_messages(tmp_path):
    # Regression: the 400 px canvas used to cut 'FLAG{4UD10_5T3G0_15_FUN}' after '15_'
    from core.audio_loader import AudioContainer
    from demo_generator import generate_multi_stego_demo

    wav = str(tmp_path / "long.wav")
    generate_multi_stego_demo(filename=wav, message="FLAG{4_VERY_LONG_SPECTROGRAM_MESSAGE_XYZ}", duration=4.0)
    c = AudioContainer(wav)
    sc = SlidingSpectrogramScanner(c.channels, c.sample_rate, c.duration, output_dir=str(tmp_path / "o"))
    sc.global_peak_mag = float(np.max(np.abs(c.channels["Mono"])))
    rendered = sc._render_window("Mono", 0.0, c.duration, 1024, int(c.sample_rate / 2 * 0.6), c.sample_rate // 2, True)
    img_contrast = rendered[0]
    # Text must end before the right edge of the track: last bright column well inside the image
    cols = np.flatnonzero((img_contrast > 200).sum(axis=0) > 3)
    assert cols.size and cols[-1] < img_contrast.shape[1] - 10


@needs_tesseract
def test_demo_sample_spectrogram_flag_is_read(tmp_path):
    from core.audio_loader import AudioContainer

    c = AudioContainer(os.path.join(SAMPLES, "sample_challenge.wav"))
    sc = SlidingSpectrogramScanner(c.channels, c.sample_rate, c.duration, output_dir=str(tmp_path), window_sec=3.5, overlap=0.5)
    _, _, top = sc.scan()
    assert "FLAG{4UD10_5T3G0_15_FUN}" in [d["decoded"] for h in top for d in h["decoded_candidates"]]
