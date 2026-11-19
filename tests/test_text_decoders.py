"""TextDecoderHub: her kodlama/şifre modu için pozitif ve negatif testler."""
import base64

import pytest

from decoders.text_decoders import FLAG_REGEX, TextDecoderHub as T


def _decoded(text):
    return [r["decoded"] for r in T.analyze_string(text)]


# ---------------- FLAG_REGEX ----------------

@pytest.mark.parametrize("flag", [
    "flag{simple}", "FLAG{UPPER_CASE}", "picoCTF{mU21C_1s_1337}", "riftCTF{Music_is_Fr33d0m}",
    "HTB_KEY{abc-123}", "SECRET{x.y.z}", "cyber{p@ss!}",
])
def test_flag_regex_matches_common_formats(flag):
    assert FLAG_REGEX.findall(f"noise {flag} noise") == [flag]


@pytest.mark.parametrize("text", ["flag{}", "flag{ab}", "flag(abc)", "just some words", "{abc}"])
def test_flag_regex_rejects_non_flags(text):
    assert not FLAG_REGEX.findall(text)


# ---------------- Tekil çözücüler ----------------

def test_base64_with_and_without_padding():
    enc = base64.b64encode(b"flag{pad}").decode()
    assert T.decode_base64_token(enc) == "flag{pad}"
    assert T.decode_base64_token(enc.rstrip("=")) == "flag{pad}"


def test_base64_rejects_binary_garbage():
    assert T.decode_base64_token(base64.b64encode(bytes(range(200, 230))).decode()) is None


def test_hex_odd_length_is_trimmed():
    assert T.decode_hex_token(b"flag{odd}".hex() + "a") == "flag{odd}"


def test_hex_rejects_non_hex():
    assert T.decode_hex_token("zzzzzz") is None


def test_rot13_is_self_inverse():
    s = "Hello, flag{World}!"
    assert T.decode_rot13(T.decode_rot13(s)) == s


@pytest.mark.parametrize("shift", [1, 3, 7, 13, 25])
def test_caesar_shift_roundtrip(shift):
    s = "Attack at Dawn {xyz}"
    assert T.caesar_shift(T.caesar_shift(s, shift), -shift) == s


def test_atbash_is_self_inverse_and_known_value():
    assert T.decode_atbash("abc XYZ") == "zyx CBA"
    assert T.decode_atbash(T.decode_atbash("flag{q}")) == "flag{q}"


def test_morse_words_and_unknown_symbols():
    assert T.decode_morse(".... ..   - .... . .-. .") == "HI THERE"
    assert T.decode_morse("") is None


@pytest.mark.parametrize("order", ["big", "little"])
def test_binary_string_both_bit_orders(order):
    data = b"flag{bits}"
    bits = "".join(format(b, "08b")[:: (1 if order == "big" else -1)] for b in data)
    assert T.decode_binary_string(bits, bitorder=order) == "flag{bits}"


def test_binary_string_too_short():
    assert T.decode_binary_string("0110") is None


def test_base32_roundtrip():
    assert T.decode_base32_token(base64.b32encode(b"flag{b32}").decode()) == "flag{b32}"
    assert T.decode_base32_token(base64.b32encode(b"flag{b32}").decode().rstrip("=")) == "flag{b32}"


def test_decimal_ascii():
    assert T.decode_decimal_ascii("72 101 108 108 111") == "Hello"
    assert T.decode_decimal_ascii("999 999 999") is None


# ---------------- analyze_string uçtan uca ----------------

@pytest.mark.parametrize("encoded, expected_type", [
    ("prefix flag{direct} suffix", "Direct Flag Match"),
    (base64.b64encode(b"flag{b64_inline}").decode(), "Base64 Decoded Flag"),
    (b"flag{hex_inline}".hex(), "Hex Decoded Flag"),
    ("synt{ebg_guvegrra}", "ROT13 Decoded Flag"),
    ("}desrever{galf", "Reversed String (Backmasking)"),
    (" ".join(format(c, "08b") for c in b"flag{bin_spaced}"), "Binary Decoded Flag (MSB-first)"),
    (base64.b32encode(b"flag{base32_inline}").decode(), "Base32 Decoded Flag"),
    (", ".join(str(c) for c in b"flag{decimal}"), "Decimal ASCII Decoded Flag"),
    (T.caesar_shift("flag{caesar_five}", 5), "Caesar Decoded Flag (shift=5)"),
    (T.decode_atbash("flag{atbash_ok}"), "Atbash Decoded Flag"),
])
def test_analyze_string_finds_each_encoding(encoded, expected_type):
    results = T.analyze_string(encoded)
    assert any(r["type"] == expected_type and r["is_flag"] for r in results), results


def test_analyze_string_base64_secret_keyword():
    token = base64.b64encode(b"the secret password is hunter2").decode()
    results = T.analyze_string(token)
    assert any(r["type"] == "Base64 Decoded Secret" for r in results)


def test_analyze_string_deduplicates():
    assert _decoded("flag{dup} flag{dup} flag{dup}").count("flag{dup}") == 1


@pytest.mark.parametrize("text", [
    "", "ab", "The quick brown fox jumps over the lazy dog.",
    "0123456789" * 20, "AAAAAAAAAAAAAAAAAAAAAAAA", "{}{}{}{}",
])
def test_analyze_string_no_false_positives(text):
    assert T.analyze_string(text) == []


def test_analyze_string_handles_long_noise_fast():
    import random
    import time

    rnd = random.Random(1)
    noise = "".join(chr(rnd.randint(32, 126)) for _ in range(30000))
    start = time.perf_counter()
    T.analyze_string(noise + " xyz{abcdef} ")
    assert time.perf_counter() - start < 2.0
