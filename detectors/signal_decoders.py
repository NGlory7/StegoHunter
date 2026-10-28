import numpy as np
from decoders.text_decoders import TextDecoderHub

DTMF_TABLE = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3", (697, 1633): "A",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6", (770, 1633): "B",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9", (852, 1633): "C",
    (941, 1209): "*", (941, 1336): "0", (941, 1477): "#", (941, 1633): "D",
}

class SignalDecoders:
    """Detects and decodes Morse code audio tones, DTMF keypad dial tones, and audio phase anomalies."""

    @staticmethod
    def detect_morse_tones(signal, sample_rate, window_ms=25):
        """
        Analyzes audio envelope to identify continuous on-off keying (Morse code beeps).
        Returns decoded text and raw morse string if found.
        """
        if len(signal) == 0:
            return None

        # Calculate short-time RMS energy
        frame_len = int(sample_rate * (window_ms / 1000.0))
        if frame_len <= 0:
            return None

        n_frames = len(signal) // frame_len
        if n_frames < 20:
            return None

        frames = signal[:n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.sqrt(np.mean(frames ** 2, axis=1))

        # Thresholding
        max_e = np.max(energy)
        min_e = np.min(energy)
        if max_e < 0.01 or (max_e - min_e) < 0.02:
            return None

        threshold = min_e + (max_e - min_e) * 0.35
        binary_envelope = (energy > threshold).astype(np.int32)

        # Run-length encoding of state durations
        runs = []
        current_state = binary_envelope[0]
        count = 1
        for val in binary_envelope[1:]:
            if val == current_state:
                count += 1
            else:
                runs.append((current_state, count))
                current_state = val
                count = 1
        runs.append((current_state, count))

        # Filter for pulse-like patterns
        active_runs = [length for state, length in runs if state == 1]
        if len(active_runs) < 4:
            return None

        min_len = np.percentile(active_runs, 20)
        dot_unit = max(1, min_len)

        morse_tokens = []
        for state, length in runs:
            if state == 1:
                # Pulse: Dot or Dash
                if length > dot_unit * 2.2:
                    morse_tokens.append("-")
                else:
                    morse_tokens.append(".")
            else:
                # Silence: intra-character (1 unit), intra-letter (3 units), or intra-word (7 units)
                if length > dot_unit * 5.0:
                    morse_tokens.append("   ")  # word space
                elif length > dot_unit * 2.0:
                    morse_tokens.append(" ")   # letter space

        morse_str = "".join(morse_tokens).strip()
        decoded_text = TextDecoderHub.decode_morse(morse_str)
        if decoded_text and len(decoded_text) >= 3:
            return {
                "type": "Morse Code Audio Tone",
                "raw_morse": morse_str,
                "decoded_text": decoded_text,
                "confidence": "HIGH" if any(c in decoded_text for c in "FLAGCTFKEY") else "MEDIUM"
            }
        return None

    @staticmethod
    def detect_pam_discrete_levels(signal):
        """
        Detects Discrete Amplitude Modulation (PAM / Sample Level Quantization Stego)
        such as picoCTF 'Surfing the Waves'.
        When samples cluster into 16 or 2 discrete levels, maps them to Hex or Binary.
        """
        # Convert float signal to int16 scale
        int16_samples = (signal * 32768.0).astype(np.int32)
        if len(int16_samples) < 50:
            return None

        for round_factor in [100, 50, 20]:
            rounded = (int16_samples // round_factor) * round_factor
            uniques, counts = np.unique(rounded, return_counts=True)
            # Filter rare single outliers (keep levels with >= 5 occurrences)
            significant = uniques[counts >= 5]

            if len(significant) == 16:
                # 16-level hex modulation
                level_sorted = sorted(significant)
                level_map = {val: f"{i:x}" for i, val in enumerate(level_sorted)}
                hex_str = "".join(level_map.get(s, "") for s in rounded if s in level_map)
                if len(hex_str) % 2 != 0:
                    hex_str = hex_str[:-1]
                try:
                    decoded_bytes = bytes.fromhex(hex_str)
                    decoded_text = decoded_bytes.decode('utf-8', errors='ignore')
                    msgs = TextDecoderHub.analyze_string(decoded_text)
                    return {
                        "type": "16-Level Amplitude Modulation (Hex PAM)",
                        "decoded_text": decoded_text,
                        "messages": msgs,
                        "confidence": "VERY HIGH" if msgs else "MEDIUM"
                    }
                except Exception:
                    pass
            elif len(significant) == 2:
                # 2-level binary modulation
                level_sorted = sorted(significant)
                bit_map = {level_sorted[0]: "0", level_sorted[1]: "1"}
                bit_str = "".join(bit_map.get(s, "") for s in rounded if s in bit_map)
                msgs = TextDecoderHub.analyze_string(bit_str)
                return {
                    "type": "2-Level Amplitude Modulation (Binary PAM)",
                    "decoded_text": bit_str[:200],
                    "messages": msgs,
                    "confidence": "HIGH"
                }
        return None

    @staticmethod
    def detect_dtmf(signal, sample_rate, window_ms=40):
        """Detects telephone DTMF tones using simplified frequency bin analysis."""
        frame_len = int(sample_rate * (window_ms / 1000.0))
        if frame_len <= 0 or len(signal) < frame_len:
            return None

        n_frames = len(signal) // frame_len
        frames = signal[:n_frames * frame_len].reshape(n_frames, frame_len)

        detected_keys = []
        last_key = None
        key_hold_count = 0

        low_freqs = [697, 770, 852, 941]
        high_freqs = [1209, 1336, 1477, 1633]

        for frame in frames:
            # Check frame RMS
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.05:
                if key_hold_count >= 2 and last_key is not None:
                    detected_keys.append(last_key)
                last_key = None
                key_hold_count = 0
                continue

            # FFT magnitude
            fft_mag = np.abs(np.fft.rfft(frame * np.hanning(frame_len)))
            freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)

            # Find peak in low group
            low_peaks = [fft_mag[np.argmin(np.abs(freqs - f))] for f in low_freqs]
            high_peaks = [fft_mag[np.argmin(np.abs(freqs - f))] for f in high_freqs]

            best_low = low_freqs[np.argmax(low_peaks)]
            best_high = high_freqs[np.argmax(high_peaks)]

            key = DTMF_TABLE.get((best_low, best_high))
            if key:
                if key == last_key:
                    key_hold_count += 1
                else:
                    if key_hold_count >= 2 and last_key is not None:
                        detected_keys.append(last_key)
                    last_key = key
                    key_hold_count = 1

        if key_hold_count >= 2 and last_key is not None:
            detected_keys.append(last_key)

        if len(detected_keys) >= 3:
            sequence = "".join(detected_keys)
            messages = []

            # 1. Decimal Integer -> Hex -> ASCII (e.g. NahamCon 'Beep Boop')
            try:
                digits_only = "".join(c for c in sequence if c.isdigit())
                if len(digits_only) >= 4:
                    num = int(digits_only)
                    hex_s = hex(num)[2:]
                    if len(hex_s) % 2 != 0:
                        hex_s = "0" + hex_s
                    raw_b = bytes.fromhex(hex_s)
                    decoded_txt = raw_b.decode('utf-8', errors='ignore')
                    txt_findings = TextDecoderHub.analyze_string(decoded_txt)
                    for tf in txt_findings:
                        messages.append(tf)
            except Exception:
                pass

            # 2. ASCII Decimal Pairs (e.g. '656667' -> 'ABC')
            try:
                digits_only = "".join(c for c in sequence if c.isdigit())
                if len(digits_only) >= 6 and len(digits_only) % 2 == 0:
                    pair_chars = []
                    for i in range(0, len(digits_only), 2):
                        val = int(digits_only[i:i+2])
                        if 32 <= val <= 126:
                            pair_chars.append(chr(val))
                    pair_txt = "".join(pair_chars)
                    if len(pair_txt) >= 4:
                        txt_findings = TextDecoderHub.analyze_string(pair_txt)
                        for tf in txt_findings:
                            messages.append(tf)
            except Exception:
                pass

            return {
                "type": "DTMF Keypad Tones",
                "sequence": sequence,
                "messages": messages,
                "confidence": "VERY HIGH" if messages else "HIGH"
            }
        return None
