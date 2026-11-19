"""SignalDecoders: sentetik Morse / DTMF / PAM / FSK / spektral bant sinyalleriyle uçtan uca testler."""
import numpy as np
import pytest

import synth
from detectors.signal_decoders import SignalDecoders as S

SR = synth.SR


def _decoded(res):
    return [m["decoded"] for m in (res or {}).get("messages", [])]


# ---------------- Morse ----------------

@pytest.mark.parametrize("text", ["SOS", "FLAG CTF", "HELLO WORLD 2026"])
def test_morse_roundtrip(text):
    res = S.detect_morse_tones(synth.morse_signal(text), SR)
    assert res and res["decoded_text"] == text


def test_morse_with_background_noise():
    sig = synth.morse_signal("STEGO HUNTER")
    sig = sig + np.random.default_rng(0).normal(0, 0.01, len(sig)).astype(np.float32)
    res = S.detect_morse_tones(sig, SR)
    assert res and res["decoded_text"] == "STEGO HUNTER"


@pytest.mark.parametrize("sig", [
    np.zeros(SR, dtype=np.float32),
    np.array([], dtype=np.float32),
    synth.tone(440, 2.0),
])
def test_morse_ignores_non_morse(sig):
    assert S.detect_morse_tones(sig, SR) is None


def test_morse_ignores_short_clicks():
    # 10 ms bursts (drum hits / spectrogram columns) are too short to be keyed Morse
    parts = []
    for i in range(30):
        parts += [synth.tone(800, 0.01), synth.silence(0.02 if i % 3 else 0.06)]
    assert S.detect_morse_tones(np.concatenate(parts), SR) is None


@pytest.mark.parametrize("text", ["EEEE TTT", "SSSS", "HI HI"])
def test_morse_edge_patterns(text):
    res = S.detect_morse_tones(synth.morse_signal(text), SR)
    assert res and res["decoded_text"] == text


# ---------------- DTMF ----------------

def test_dtmf_full_keypad():
    keys = "123A456B789C*0#D"
    res = S.detect_dtmf(synth.dtmf_signal(keys), SR)
    assert res and res["sequence"] == keys


def test_dtmf_bigint_flag():
    digits = str(int.from_bytes(b"flag{dtmf}", "big"))
    res = S.detect_dtmf(synth.dtmf_signal(digits), SR)
    assert res["sequence"] == digits
    assert "flag{dtmf}" in _decoded(res)


def test_dtmf_decimal_pairs_flag():
    # 'FLAG{DTMF}' -> mixed-width ASCII decimal (2 and 3 digit codes)
    digits = "".join(str(ord(c)) for c in "FLAG{DTMF}")
    res = S.detect_dtmf(synth.dtmf_signal(digits), SR)
    assert "FLAG{DTMF}" in _decoded(res)


def test_dtmf_repeated_key_needs_gap():
    res = S.detect_dtmf(synth.dtmf_signal("1111"), SR)
    assert res["sequence"] == "1111"


def test_dtmf_silence_returns_none():
    assert S.detect_dtmf(np.zeros(SR, dtype=np.float32), SR) is None


# ---------------- PAM ----------------

def test_pam16_hex_levels():
    payload = bytes(range(256)) + b"\n flag{pam16_ok} \n"
    res = S.detect_pam_discrete_levels(synth.pam16_signal(payload))
    assert res["type"].startswith("16-Level")
    assert "flag{pam16_ok}" in _decoded(res)


def test_pam2_binary_levels_now_decode():
    res = S.detect_pam_discrete_levels(synth.pam2_signal(b"xx flag{pam_binary} xx"))
    assert res["type"].startswith("2-Level")
    assert "flag{pam_binary}" in _decoded(res)


def test_pam_ignores_continuous_audio():
    assert S.detect_pam_discrete_levels(synth.tone(440, 1.0) + synth.tone(523, 1.0, 0.2)) is None


# ---------------- FSK ----------------

@pytest.mark.parametrize("baud, f0, f1", [(100, 1200, 2200), (50, 1000, 2000), (200, 1500, 3000)])
def test_fsk_roundtrip(baud, f0, f1):
    res = S.detect_fsk(synth.fsk_signal(b"flag{fsk_modem}", baud=baud, f0=f0, f1=f1), SR)
    assert res, "FSK not detected"
    assert "flag{fsk_modem}" in _decoded(res)
    assert abs(res["baud_estimate"] - baud) / baud < 0.1
    lo, hi = res["mark_space_hz"]
    assert abs(lo - f0) < 100 and abs(hi - f1) < 100


def test_fsk_plain_text_without_flag():
    res = S.detect_fsk(synth.fsk_signal(b"hello modem world"), SR)
    assert res and res["decoded_text"].startswith("hello modem world")
    assert res["messages"] == []


@pytest.mark.parametrize("sig", [
    synth.tone(440, 2.0),
    np.zeros(SR, dtype=np.float32),
    synth.morse_signal("SOS"),
])
def test_fsk_rejects_non_fsk(sig):
    assert S.detect_fsk(sig, SR) is None


def test_fsk_rejects_white_noise():
    noise = np.random.default_rng(3).normal(0, 0.3, SR * 2).astype(np.float32)
    assert S.detect_fsk(noise, SR) is None


# ---------------- Spektral bantlar ----------------

def test_spectral_clean_tone_is_not_suspicious():
    res = S.analyze_spectral_bands(synth.tone(440, 2.0), SR)
    assert not res["is_suspicious"]
    assert res["band_ratios"]["low_mid_250-2k"] > 0.95


def test_spectral_ultrasonic_carrier_is_flagged():
    sig = synth.tone(440, 2.0, 0.3) + synth.tone(18000, 2.0, 0.2)
    res = S.analyze_spectral_bands(sig, SR)
    assert res["is_suspicious"]
    assert abs(res["high_band_peak_hz"] - 18000) < 50


def test_spectral_low_sample_rate_is_handled():
    res = S.analyze_spectral_bands(synth.tone(440, 1.0, sr=8000), 8000)
    assert res["is_suspicious"] is False


def test_spectral_ratios_sum_to_one():
    sig = np.random.default_rng(1).normal(0, 0.2, SR).astype(np.float32)
    res = S.analyze_spectral_bands(sig, SR)
    assert sum(res["band_ratios"].values()) == pytest.approx(1.0, abs=0.01)


# ---------------- DTMF yanlış pozitif koruması ----------------

@pytest.mark.parametrize("sig", [
    synth.tone(1209, 1.0),                                  # single DTMF-group tone, no partner
    synth.fsk_signal(b"flag{not_dtmf}"),                    # 1200 Hz sits right next to 1209 Hz
    synth.morse_signal("SOS", freq=770),                    # low-group tone alone
    np.random.default_rng(5).normal(0, 0.3, SR).astype(np.float32),
    synth.tone(440, 1.0) + synth.tone(3000, 1.0),           # two tones, neither DTMF
])
def test_dtmf_rejects_non_dtmf_audio(sig):
    assert S.detect_dtmf(sig, SR) is None


def test_dtmf_survives_moderate_noise():
    sig = synth.dtmf_signal("0123456789")
    sig = sig + np.random.default_rng(2).normal(0, 0.02, len(sig)).astype(np.float32)
    assert S.detect_dtmf(sig, SR)["sequence"] == "0123456789"


# ---------------- Ton bölgeleri (karışık sinyaller) ----------------

def _morse_then_fsk(gap=0.5):
    return np.concatenate([synth.morse_signal("CTF FLAG"), synth.silence(gap), synth.fsk_signal(b"flag{fsk_after_morse}")])


def test_tone_regions_split_morse_and_fsk():
    regions = S.find_tone_regions(_morse_then_fsk(), SR)
    assert len(regions) == 2
    (a0, a1), (b0, b1) = regions
    assert a1 <= b0, "regions must not overlap"


def test_scan_tone_regions_decodes_both_transmissions():
    res = S.scan_tone_regions(_morse_then_fsk(), SR)
    assert [m["decoded_text"] for m in res["morse"]] == ["CTF FLAG"]
    assert "flag{fsk_after_morse}" in [m["decoded"] for f in res["fsk"] for m in f["messages"]]
    assert res["fsk"][0]["region"][0] > res["morse"][0]["region"][1] - 0.1


def test_scan_tone_regions_fsk_then_morse_order():
    sig = np.concatenate([synth.fsk_signal(b"flag{fsk_first}"), synth.silence(0.6), synth.morse_signal("SOS")])
    res = S.scan_tone_regions(sig, SR)
    assert [m["decoded_text"] for m in res["morse"]] == ["SOS"]
    assert "flag{fsk_first}" in [m["decoded"] for f in res["fsk"] for m in f["messages"]]


def test_scan_tone_regions_two_morse_messages_on_different_pitches():
    sig = np.concatenate([synth.morse_signal("ONE", freq=600), synth.silence(0.8), synth.morse_signal("TWO", freq=1500)])
    res = S.scan_tone_regions(sig, SR)
    text = " ".join(m["decoded_text"] for m in res["morse"]).split()
    assert text == ["ONE", "TWO"]


def test_scan_tone_regions_single_transmission_is_not_split():
    res = S.scan_tone_regions(synth.morse_signal("HELLO WORLD"), SR)
    assert [m["decoded_text"] for m in res["morse"]] == ["HELLO WORLD"]
    assert len(res["regions"]) == 1


def test_find_tone_regions_silence_and_empty():
    assert S.find_tone_regions(np.zeros(SR, dtype=np.float32), SR) == []
    assert S.find_tone_regions(np.array([], dtype=np.float32), SR) == []
