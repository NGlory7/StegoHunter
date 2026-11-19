"""
Deterministic synthetic test-signal generators.
Every stego vector the suite claims to detect gets a tiny, reproducible fixture here,
so tests don't depend on large third-party sample files.
"""
import struct
import wave

import numpy as np

SR = 44100

MORSE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
}

DTMF_FREQS = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}


def tone(freq, seconds, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def morse_signal(text, unit=0.06, freq=700, sr=SR):
    parts = [silence(unit * 3, sr)]
    for wi, word in enumerate(text.upper().split()):
        if wi:
            parts.append(silence(unit * 7, sr))
        for li, ch in enumerate(word):
            if li:
                parts.append(silence(unit * 3, sr))
            for si, sym in enumerate(MORSE[ch]):
                if si:
                    parts.append(silence(unit, sr))
                parts.append(tone(freq, unit * (3 if sym == '-' else 1), sr=sr))
    parts.append(silence(unit * 3, sr))
    return np.concatenate(parts)


def dtmf_signal(keys, key_s=0.12, gap_s=0.08, sr=SR):
    parts = []
    for k in keys:
        lo, hi = DTMF_FREQS[k]
        parts.append(tone(lo, key_s, 0.35, sr) + tone(hi, key_s, 0.35, sr))
        parts.append(silence(gap_s, sr))
    return np.concatenate(parts)


def fsk_signal(payload, baud=100, f0=1200, f1=2200, sr=SR, lead_s=0.05):
    bits = "".join(format(b, "08b") for b in payload)
    spb = sr // baud
    t = np.arange(len(bits) * spb) / sr
    freqs = np.repeat([f1 if b == "1" else f0 for b in bits], spb)
    phase = 2 * np.pi * np.cumsum(freqs) / sr  # continuous-phase FSK
    sig = (0.5 * np.sin(phase)).astype(np.float32)
    assert len(t) == len(sig)
    return np.concatenate([silence(lead_s, sr), sig, silence(lead_s, sr)])


def pam16_signal(payload, step=2000, repeat=1):
    """Each hex nibble -> one of 16 equally spaced int16 levels."""
    nibbles = [int(c, 16) for c in payload.hex()]
    levels = np.array([(-15 + 2 * n) * step // 2 for n in nibbles], dtype=np.float32)
    return np.repeat(levels / 32768.0, repeat)


def pam2_signal(payload, level=8000, repeat=1):
    bits = "".join(format(b, "08b") for b in payload)
    vals = np.array([level if b == "1" else -level for b in bits], dtype=np.float32)
    return np.repeat(vals / 32768.0, repeat)


def write_wav(path, samples, sr=SR, channels=1, sampwidth=2):
    """Writes float samples in [-1, 1] (interleaved if channels>1) as integer PCM."""
    samples = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    if sampwidth == 1:
        data = ((samples * 127) + 128).astype(np.uint8).tobytes()
    elif sampwidth == 2:
        data = (samples * 32767).astype("<i2").tobytes()
    elif sampwidth == 3:
        ints = (samples * 8388607).astype(np.int32)
        b = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        data = b.tobytes()
    elif sampwidth == 4:
        data = (samples * 2147483647).astype("<i4").tobytes()
    else:
        raise ValueError(sampwidth)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(data)
    return path


def write_int16_wav(path, int_samples, sr=SR, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(int_samples, dtype="<i2").tobytes())
    return path


def build_riff(chunks, fmt_tag=1, channels=1, sr=SR, bits=16, extensible_subformat=None, riff_size=None):
    """
    Hand-builds a RIFF/WAVE file from (id, payload) chunk tuples, inserting the fmt chunk first.
    Allows float / WAVE_FORMAT_EXTENSIBLE / bogus header sizes that the stdlib `wave` refuses.
    """
    block_align = channels * bits // 8
    if extensible_subformat is not None:
        guid_tail = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, sr, sr * block_align, block_align, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", extensible_subformat) + guid_tail
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, channels, sr, sr * block_align, block_align, bits)
    body = b"WAVE"
    for cid, payload in [(b"fmt ", fmt)] + list(chunks):
        body += cid + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")
    size = len(body) if riff_size is None else riff_size
    return b"RIFF" + struct.pack("<I", size) + body


def info_list_chunk(fields):
    """Builds a LIST/INFO payload from {'INAM': 'title', 'ICMT': 'comment', ...}."""
    payload = b"INFO"
    for key, val in fields.items():
        raw = val.encode() + b"\x00"
        payload += key.encode() + struct.pack("<I", len(raw)) + raw + (b"\x00" if len(raw) % 2 else b"")
    return (b"LIST", payload)


def embed_lsb(cover_int16, payload, bit=0):
    """Writes payload bits (MSB-first per byte) into bit `bit` of consecutive samples."""
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    out = np.array(cover_int16, dtype=np.int16).copy()
    assert len(bits) <= len(out), "cover too short"
    mask = np.int16(~(1 << bit))
    out[: len(bits)] = (out[: len(bits)] & mask) | (bits.astype(np.int16) << bit)
    return out


def noisy_cover(n, seed=0, amp=3000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    base = amp * np.sin(2 * np.pi * 330 * t) + rng.normal(0, amp * 0.05, n)
    return base.astype(np.int16)
