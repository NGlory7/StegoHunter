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
    def _two_means(values, iterations=20):
        """1-D k-means with k=2. Returns (low_centre, high_centre)."""
        vals = np.asarray(values, dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        for _ in range(iterations):
            split = (lo + hi) / 2.0
            low, high = vals[vals <= split], vals[vals > split]
            if len(low) == 0 or len(high) == 0:
                break
            new_lo, new_hi = float(low.mean()), float(high.mean())
            if new_lo == lo and new_hi == hi:
                break
            lo, hi = new_lo, new_hi
        return lo, hi

    @staticmethod
    def detect_morse_tones(signal, sample_rate, window_ms=10):
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

        # Envelope smearing makes pulses look longer and gaps shorter than their nominal
        # unit lengths, so dots/dashes are separated by 1-D 2-means clustering and gap
        # thresholds are derived from those cluster centres (dot ~ 1 unit, dash ~ 3 units).
        dot_len, dash_len = SignalDecoders._two_means(active_runs)
        if dash_len < dot_len * 1.8:
            # Only one pulse length present (e.g. 'EEE' or 'TTT'): treat all as dots
            dot_len = float(np.median(active_runs))
            dash_len = dot_len * 3.0
        # Plausibility: even 40 WPM Morse has 30 ms dots; shorter "pulses" are clicks,
        # drum hits or spectrogram-art columns, not keyed tones.
        if dot_len * window_ms < 25:
            return None
        pulse_split = (dot_len + dash_len) / 2.0
        letter_gap = (dot_len + dash_len) / 2.0
        word_gap = dash_len * 1.6

        morse_tokens = []
        for idx, (state, length) in enumerate(runs):
            if state == 1:
                morse_tokens.append("-" if length > pulse_split else ".")
            elif 0 < idx < len(runs) - 1:
                # Silence: intra-character (1 unit), letter (3 units) or word (7 units)
                if length > word_gap:
                    morse_tokens.append("   ")
                elif length > letter_gap:
                    morse_tokens.append(" ")

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

            # 3. Mixed-width ASCII decimal ('70 76 65 71 123 ...' typed without separators)
            var_txt = TextDecoderHub.decode_concatenated_decimal("".join(c for c in sequence if c.isdigit()))
            if var_txt:
                messages.extend(TextDecoderHub.analyze_string(var_txt))
            messages = TextDecoderHub._dedupe(messages)

            return {
                "type": "DTMF Keypad Tones",
                "sequence": sequence,
                "messages": messages,
                "confidence": "VERY HIGH" if messages else "HIGH"
            }
        return None

    @staticmethod
    def _dominant_freq_track(signal, sample_rate, win_ms=5.0, hop_ms=1.0, min_rms=0.02):
        """Returns per-hop dominant frequency (Hz) and an 'active' mask (frame RMS above floor)."""
        win = max(32, int(sample_rate * win_ms / 1000.0))
        hop = max(1, int(sample_rate * hop_ms / 1000.0))
        if len(signal) < win * 4:
            return None, None
        n_frames = 1 + (len(signal) - win) // hop
        idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = signal[idx].astype(np.float32)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        n_fft = 1 << int(np.ceil(np.log2(win * 4)))  # zero-pad for finer frequency bins
        mag = np.abs(np.fft.rfft(frames * np.hanning(win)[None, :], n=n_fft, axis=1))
        mag[:, 0] = 0
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        dom = freqs[np.argmax(mag, axis=1)]
        return dom, rms > min_rms

    @staticmethod
    def detect_fsk(signal, sample_rate, max_seconds=120):
        """
        Detects binary FSK (two alternating carrier tones, e.g. Bell 202 / modem / 'beep-bop' bits).
        Tracks the dominant frequency every 1 ms, verifies exactly two tone clusters dominate,
        estimates the bit duration from run lengths and decodes the bitstream as text.
        """
        if len(signal) == 0 or sample_rate <= 0:
            return None
        signal = signal[: int(sample_rate * max_seconds)]
        dom, active = SignalDecoders._dominant_freq_track(signal, sample_rate)
        if dom is None or active.sum() < 40:
            return None

        act = dom[active]
        # Two-cluster split around the median of active tones
        split = float(np.median(act))
        low, high = act[act < split], act[act >= split]
        if len(low) < 10 or len(high) < 10:
            # Median can land exactly on one tone - fall back to midpoint of extremes
            split = (float(np.percentile(act, 5)) + float(np.percentile(act, 95))) / 2.0
            low, high = act[act < split], act[act >= split]
            if len(low) < 10 or len(high) < 10:
                return None
        f0, f1 = float(np.median(low)), float(np.median(high))
        if f1 - f0 < 150:
            return None
        # Tones must be stable: most frames within 15% of their cluster centre
        tol0, tol1 = max(60.0, 0.15 * f0), max(60.0, 0.15 * f1)
        stable = (np.abs(low - f0) < tol0).mean() > 0.85 and (np.abs(high - f1) < tol1).mean() > 0.85
        if not stable:
            return None

        # Symbols: 0 (low tone), 1 (high tone), -1 (silence)
        symbols = np.where(active, (dom >= split).astype(np.int8), -1)
        runs = []
        cur, count = symbols[0], 1
        for v in symbols[1:]:
            if v == cur:
                count += 1
            else:
                runs.append((int(cur), count))
                cur, count = v, 1
        runs.append((int(cur), count))

        tone_runs = [c for s, c in runs if s != -1 and c >= 3]
        if len(tone_runs) < 8:
            return None
        unit = float(np.percentile(tone_runs, 10))
        # Refine: average of runs that look like single bits
        singles = [c for c in tone_runs if c < unit * 1.5]
        if singles:
            unit = float(np.mean(singles))
        if unit < 3:
            return None

        bit_chunks, current = [], []
        for s, c in runs:
            if s == -1:
                if current:
                    bit_chunks.append("".join(current))
                    current = []
                continue
            n_bits = int(round(c / unit))
            current.append(str(s) * n_bits)
        if current:
            bit_chunks.append("".join(current))
        bits = "".join(bit_chunks)
        if len(bits) < 24:
            return None

        messages = []
        decoded_text = None
        for order in ("big", "little"):
            txt = TextDecoderHub.decode_binary_string(bits, bitorder=order)
            if txt:
                decoded_text = decoded_text or txt
                messages.extend(TextDecoderHub.analyze_string(txt))
        messages.extend(TextDecoderHub.analyze_string(bits))
        messages = TextDecoderHub._dedupe(messages)

        return {
            "type": "Binary FSK (Two-Tone Frequency Shift Keying)",
            "mark_space_hz": (round(f0), round(f1)),
            "baud_estimate": round(1000.0 / unit, 1),
            "bitstream": bits[:512],
            "decoded_text": decoded_text,
            "messages": messages,
            "confidence": "VERY HIGH" if messages else ("HIGH" if decoded_text else "MEDIUM"),
        }

    @staticmethod
    def analyze_spectral_bands(signal, sample_rate, high_cutoff=15000.0, threshold=0.08):
        """
        Measures how much energy sits in the near-ultrasonic band (>= high_cutoff Hz).
        Natural recordings rarely carry much energy there; spectrogram text, hidden
        carriers and watermarks usually do. Returns band ratios and a suspicion flag.
        """
        if len(signal) < 1024 or sample_rate <= 0:
            return None
        nyquist = sample_rate / 2.0
        if high_cutoff >= nyquist:
            return {"high_band_ratio": 0.0, "is_suspicious": False, "note": "Sample rate too low for high-band analysis"}

        n_fft = 4096
        n_frames = len(signal) // n_fft
        frames = signal[: n_frames * n_fft].reshape(n_frames, n_fft)
        if n_frames > 400:  # subsample long files evenly
            frames = frames[np.linspace(0, n_frames - 1, 400).astype(int)]
        power = (np.abs(np.fft.rfft(frames * np.hanning(n_fft)[None, :], axis=1)) ** 2).sum(axis=0)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        total = float(power[1:].sum())
        if total <= 0:
            return None

        bands = {
            "sub_bass_<250Hz": (0, 250),
            "low_mid_250-2k": (250, 2000),
            "high_mid_2k-8k": (2000, 8000),
            "presence_8k-15k": (8000, high_cutoff),
            "ultrasonic_>=15k": (high_cutoff, nyquist + 1),
        }
        ratios = {name: round(float(power[(freqs >= lo) & (freqs < hi)].sum()) / total, 4) for name, (lo, hi) in bands.items()}
        high_ratio = ratios["ultrasonic_>=15k"]

        # Peak frequency inside the high band (where a hidden carrier would sit)
        hb = freqs >= high_cutoff
        peak_hz = float(freqs[hb][np.argmax(power[hb])]) if hb.any() else 0.0

        return {
            "type": "Spectral Band Energy Profile",
            "band_ratios": ratios,
            "high_band_ratio": high_ratio,
            "high_band_peak_hz": round(peak_hz),
            "is_suspicious": high_ratio >= threshold,
        }
