import os
import struct
import numpy as np
from core.crypto_hasher import calculate_file_hashes

class AudioLoadError(Exception):
    pass

class AudioContainer:
    """Holds parsed audio channels, metadata, chunks, and trailing slack payload."""
    def __init__(self, filepath):
        self.filepath = os.path.abspath(filepath)
        self.filename = os.path.basename(filepath)
        self.filesize = os.path.getsize(filepath)
        self.hashes = calculate_file_hashes(filepath)
        
        self.sample_rate = 44100
        self.num_channels = 1
        self.sampwidth = 2
        self.duration = 0.0
        
        self.channels = {}  # e.g. {"Left": np.ndarray, "Right": np.ndarray, ...}
        self.metadata_chunks = []
        self.appended_payload = b""
        self.declared_riff_size = 0
        self.declared_data_size = 0
        
        self._parse_wav_structure()

    def _parse_wav_structure(self):
        """
        Manually parses RIFF WAV headers to safely extract:
        1. Audio sample data (supports 8, 16, 24, 32-bit PCM)
        2. RIFF chunks (INFO, LIST, ID3, JUNK, etc.)
        3. Trailing EOF payload (data appended after the official audio data chunk)
        """
        with open(self.filepath, "rb") as f:
            raw_bytes = f.read()

        if len(raw_bytes) < 44 or raw_bytes[0:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
            raise AudioLoadError(f"File '{self.filename}' is not a valid RIFF/WAVE file!")

        self.declared_riff_size = struct.unpack("<I", raw_bytes[4:8])[0] + 8

        offset = 12
        audio_data_bytes = None
        audio_format = 1  # 1 = PCM

        while offset + 8 <= len(raw_bytes):
            chunk_id = raw_bytes[offset:offset+4]
            chunk_size = struct.unpack("<I", raw_bytes[offset+4:offset+8])[0]
            chunk_data_start = offset + 8
            chunk_data_end = chunk_data_start + chunk_size

            # Safety guard against corrupted or huge chunk size
            if chunk_data_end > len(raw_bytes):
                # Chunk extends beyond file boundary
                actual_chunk_data = raw_bytes[chunk_data_start:]
            else:
                actual_chunk_data = raw_bytes[chunk_data_start:chunk_data_end]

            if chunk_id == b"fmt ":
                if len(actual_chunk_data) >= 16:
                    audio_format, self.num_channels, self.sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack("<HHIIHH", actual_chunk_data[:16])
                    self.sampwidth = bits_per_sample // 8
            elif chunk_id == b"data":
                self.declared_data_size = chunk_size
                audio_data_bytes = actual_chunk_data
                # Everything strictly after this audio chunk (plus word padding) might be trailing payload
                expected_end = chunk_data_end + (1 if chunk_size % 2 != 0 else 0)
                if len(raw_bytes) > expected_end:
                    self.appended_payload = raw_bytes[expected_end:]
            else:
                # Other chunks: LIST, INFO, ID3, JUNK, etc.
                text_content = ""
                try:
                    # Filter printable characters
                    printable = bytes([b for b in actual_chunk_data if 32 <= b <= 126 or b in (10, 13)]).decode('ascii', errors='ignore').strip()
                    if len(printable) > 3:
                        text_content = printable
                except Exception:
                    pass

                self.metadata_chunks.append({
                    "id": chunk_id.decode('ascii', errors='replace'),
                    "size": chunk_size,
                    "offset": offset,
                    "text_content": text_content
                })

            # RIFF chunks are word-aligned (even byte boundary)
            padded_size = chunk_size + (1 if chunk_size % 2 != 0 else 0)
            offset += 8 + padded_size

        if audio_data_bytes is None:
            raise AudioLoadError("No 'data' chunk found in WAV file!")

        self._decode_pcm_samples(audio_data_bytes)

    def _decode_pcm_samples(self, raw_audio):
        """Converts raw byte stream into normalized float32 channels (-1.0 to 1.0)."""
        if self.sampwidth == 1:
            samples = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif self.sampwidth == 2:
            samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif self.sampwidth == 3:
            raw_arr = np.frombuffer(raw_audio, dtype=np.uint8)
            n_samples = len(raw_arr) // 3
            raw_arr = raw_arr[:n_samples * 3].reshape(-1, 3)
            samples32 = (raw_arr[:, 0].astype(np.int32) |
                         (raw_arr[:, 1].astype(np.int32) << 8) |
                         (raw_arr[:, 2].astype(np.int32) << 16))
            sign_mask = 1 << 23
            samples32 = (samples32 ^ sign_mask) - sign_mask
            samples = samples32.astype(np.float32) / 8388608.0
        elif self.sampwidth == 4:
            samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise AudioLoadError(f"Unsupported sample width: {self.sampwidth} bytes ({self.sampwidth * 8} bits)")

        total_frames = len(samples) // self.num_channels
        samples = samples[:total_frames * self.num_channels]
        self.duration = total_frames / float(self.sample_rate)

        if self.num_channels > 1:
            reshaped = samples.reshape(-1, self.num_channels)
            left = reshaped[:, 0]
            right = reshaped[:, 1]
            diff = left - right
            mono_mix = (left + right) / 2.0

            self.channels["Channel_Left"] = left
            self.channels["Channel_Right"] = right
            self.channels["Channel_Diff_LR"] = diff
            self.channels["Channel_Mono"] = mono_mix
        else:
            self.channels["Mono"] = samples
