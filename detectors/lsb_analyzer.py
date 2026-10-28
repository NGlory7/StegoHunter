import os
import math
import wave
import numpy as np
import cv2
from decoders.text_decoders import TextDecoderHub, FLAG_REGEX

class LSBAnalyzer:
    """
    Analyzes Least Significant Bits (LSB 0, 1, 2), calculates Shannon Entropy,
    renders 2D Bit-Plane matrix images, and scans for hidden flags/files across bitstream variations
    (LSB-first / MSB-first, bit-shifts, stereo channel isolation).
    """
    def __init__(self, raw_audio_path, output_dir="stego_output"):
        self.raw_audio_path = raw_audio_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []

    @staticmethod
    def calculate_shannon_entropy(byte_array):
        """Calculates Shannon entropy in bits per byte (0.0 to 8.0)."""
        if len(byte_array) == 0:
            return 0.0
        counts = np.bincount(byte_array, minlength=256)
        probs = counts[counts > 0] / float(len(byte_array))
        return float(-np.sum(probs * np.log2(probs)))

    def _extract_samples(self):
        """Extracts audio samples as numpy array handling 8-bit, 16-bit, and 32-bit PCM."""
        try:
            with wave.open(self.raw_audio_path, "rb") as w:
                n_channels = w.getnchannels()
                sampwidth = w.getsampwidth()
                frames = w.readframes(w.getnframes())
                if sampwidth == 2:
                    samples = np.frombuffer(frames, dtype=np.int16)
                elif sampwidth == 1:
                    samples = np.frombuffer(frames, dtype=np.uint8)
                elif sampwidth == 4:
                    samples = np.frombuffer(frames, dtype=np.int32)
                else:
                    samples = np.frombuffer(frames, dtype=np.int16)
                return samples, n_channels
        except Exception:
            pass

        # Fallback to direct raw parsing
        with open(self.raw_audio_path, "rb") as f:
            raw = f.read()
        idx = raw.find(b"data")
        if idx != -1 and idx + 8 < len(raw):
            data_bytes = raw[idx+8:]
            n_samples = len(data_bytes) // 2
            samples = np.frombuffer(data_bytes[:n_samples * 2], dtype=np.int16)
            return samples, 2
        return np.array([], dtype=np.int16), 1

    def analyze(self):
        """Analyzes bit planes across all channels, bit orders, and bit shifts."""
        samples, n_channels = self._extract_samples()
        if len(samples) < 50:
            return self.results

        # Define channel views to test
        channel_configs = [("All Channels", samples)]
        if n_channels == 2 and len(samples) >= 4:
            channel_configs.append(("Left Channel", samples[0::2]))
            channel_configs.append(("Right Channel", samples[1::2]))

        for bit_idx in [0, 1, 2]:
            all_bits = ((samples >> bit_idx) & 1).astype(np.uint8)

            # Standard baseline entropy (MSB first, all samples)
            padded_len = (len(all_bits) // 8) * 8
            if padded_len == 0:
                continue

            bit_slice = all_bits[:padded_len].reshape(-1, 8)
            packed_default = np.packbits(bit_slice, axis=-1, bitorder='big').flatten()
            entropy = self.calculate_shannon_entropy(packed_default[:50000])

            # Generate 2D Bit-plane visualizer image (e.g. 512x512)
            side = int(math.isqrt(min(len(all_bits), 262144)))
            plane_img_rel = None
            if side >= 128:
                grid = (all_bits[:side * side] * 255).reshape(side, side)
                plane_img_name = f"lsb_bitplane_bit{bit_idx}.png"
                plane_img_path = os.path.join(self.output_dir, plane_img_name)
                grid_resized = cv2.resize(grid, (512, 512), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(plane_img_path, grid_resized)
                plane_img_rel = plane_img_name

            # Comprehensive bitstream decoding scan across variations
            detected_messages = []
            seen_decoded_texts = set()

            for ch_label, ch_samples in channel_configs:
                ch_bits = ((ch_samples >> bit_idx) & 1).astype(np.uint8)
                for bitorder in ['little', 'big']:
                    # Test shifts 0 to 7
                    for shift in range(8):
                        s_bits = ch_bits[shift:]
                        plen = (len(s_bits) // 8) * 8
                        if plen < 64:
                            continue
                        packed = np.packbits(s_bits[:plen].reshape(-1, 8), axis=-1, bitorder=bitorder).flatten()
                        # Examine first 30,000 bytes
                        prefix_bytes = packed[:30000].tobytes()
                        # Clean printable ASCII
                        printable = "".join(chr(b) if (32 <= b <= 126 or b in (10, 13)) else ' ' for b in prefix_bytes)
                        findings = TextDecoderHub.analyze_string(printable)
                        for f in findings:
                            txt = f.get("decoded", "")
                            if txt and txt not in seen_decoded_texts:
                                seen_decoded_texts.add(txt)
                                f_copy = dict(f)
                                f_copy["variant"] = f"{ch_label}, Bit {bit_idx}, order={bitorder}, shift={shift}"
                                detected_messages.append(f_copy)

            has_flag = any(f.get("is_flag") for f in detected_messages)
            is_suspicious_entropy = entropy > 7.92

            self.results.append({
                "bit_index": bit_idx,
                "entropy": round(entropy, 3),
                "is_suspicious_entropy": is_suspicious_entropy,
                "has_flag": has_flag,
                "decoded_messages": detected_messages,
                "bitplane_img": plane_img_rel,
                "sample_count": len(samples)
            })

        return self.results
